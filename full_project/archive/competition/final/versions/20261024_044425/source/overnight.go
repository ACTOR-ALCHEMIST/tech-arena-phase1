// Run with: .tools/go/bin/go run overnight.go -max-experiments 30
// Local orchestration, resumable result files, live logs and retrieval-guided plans.
package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"sync"
	"syscall"
	"time"
)

type Plan struct {
	Name   string   `json:"name"`
	Argv   []string `json:"argv"`
	Reason string   `json:"reason"`
	Done   bool     `json:"done"`
}

func writeJSON(path string, value any) {
	os.MkdirAll(filepath.Dir(path), 0755)
	b, _ := json.MarshalIndent(value, "", "  ")
	os.WriteFile(path+".tmp", b, 0644)
	os.Rename(path+".tmp", path)
}

func journal(kind, message string) {
	zone := time.FixedZone("CST", 8*3600)
	b, _ := json.Marshal(map[string]any{"time": time.Now().In(zone).Format(time.RFC3339), "kind": kind, "message": message})
	f, err := os.OpenFile("competition/timeline.jsonl", os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0644)
	if err == nil {
		defer f.Close()
		f.Write(append(b, '\n'))
	}
}

func lockLane(ctx context.Context, path string) (*os.File, error) {
	file, err := os.OpenFile(path, os.O_CREATE|os.O_RDWR, 0644)
	if err != nil {
		return nil, err
	}
	for {
		err = syscall.Flock(int(file.Fd()), syscall.LOCK_EX|syscall.LOCK_NB)
		if err == nil {
			return file, nil
		}
		if err != syscall.EWOULDBLOCK && err != syscall.EAGAIN && err != syscall.EINTR {
			file.Close()
			return nil, err
		}
		select {
		case <-ctx.Done():
			file.Close()
			return nil, ctx.Err()
		case <-time.After(15 * time.Second):
		}
	}
}

func runLane(ctx context.Context, python, lane string, maximum int, env []string) int {
	attempted := 0
	lock, err := lockLane(ctx, filepath.Join("competition", ".lane_"+lane+".lock"))
	if err != nil {
		journal("scheduler", lane+" 通道锁等待结束："+err.Error())
		return attempted
	}
	defer lock.Close()
	statePath := filepath.Join("competition", "controller_"+lane+".json")
	if b, err := os.ReadFile(statePath); err == nil {
		var old map[string]any
		if json.Unmarshal(b, &old) == nil && (old["state"] == "running" || old["state"] == "waiting") {
			if value, ok := old["controller_pid"].(float64); ok && int(value) != os.Getpid() {
				proc, _ := os.FindProcess(int(value))
				if proc != nil && proc.Signal(syscall.Signal(0)) == nil {
					journal("scheduler", lane+" 通道已有存活调度进程；等待其结束后恢复剩余队列，避免重复训练。")
					for proc.Signal(syscall.Signal(0)) == nil {
						select {
						case <-ctx.Done():
							return attempted
						case <-time.After(15 * time.Second):
						}
						if current, err := os.ReadFile(statePath); err == nil {
							var value map[string]any
							if json.Unmarshal(current, &value) == nil && value["state"] == "finished" {
								break
							}
						}
					}
				}
			}
		}
	}
	if lane == "gpu" {
		health := exec.CommandContext(ctx, python, "-c", "import torch; assert torch.backends.mps.is_available(); x=torch.zeros(1,device='mps'); torch.mps.synchronize(); print('MPS allocation healthy')")
		health.Env = env
		if b, err := health.CombinedOutput(); err != nil {
			journal("failure", "GPU 通道健康检查失败，停止本通道以避免重复失败："+string(b))
			return attempted
		}
	}
	for i := 0; i < maximum; i++ {
		if ctx.Err() != nil {
			break
		}
		planner := exec.CommandContext(ctx, python, "arena_plan.py", "next", "--lane", lane)
		planner.Env = env
		output, err := planner.Output()
		if err != nil {
			journal("failure", "实验规划失败："+err.Error())
			break
		}
		var plan Plan
		if err = json.Unmarshal(output, &plan); err != nil {
			journal("failure", "规划 JSON 无效："+string(output))
			break
		}
		if plan.Done {
			otherLane := "cpu"
			if lane == "cpu" {
				otherLane = "gpu"
			}
			otherPath := filepath.Join("competition", "controller_"+otherLane+".json")
			if b, readErr := os.ReadFile(otherPath); readErr == nil {
				var other map[string]any
				if json.Unmarshal(b, &other) == nil && other["state"] == "running" {
					if pid, ok := other["controller_pid"].(float64); ok {
						process, _ := os.FindProcess(int(pid))
						if process != nil && process.Signal(syscall.Signal(0)) == nil {
							writeJSON(statePath, map[string]any{"controller_pid": os.Getpid(), "lane": lane, "state": "waiting", "waiting_for": otherLane})
							select {
							case <-ctx.Done():
							case <-time.After(15 * time.Second):
							}
							i-- // Waiting for dependencies does not consume an experiment.
							continue
						}
					}
				}
			}
			journal("scheduler", plan.Reason)
			break
		}
		attempted++
		logPath := filepath.Join("competition/logs", plan.Name+".log")
		log, err := os.Create(logPath)
		if err != nil {
			panic(err)
		}
		journal("retrieval", plan.Name+": "+plan.Reason)
		jobCtx, jobCancel := context.WithTimeout(ctx, 30*time.Minute)
		cmd := exec.CommandContext(jobCtx, python, append([]string{"-u"}, plan.Argv...)...)
		cmd.Env = env
		cmd.Stdout = io.MultiWriter(os.Stdout, log)
		cmd.Stderr = io.MultiWriter(os.Stderr, log)
		started := time.Now()
		if err = cmd.Start(); err == nil {
			writeJSON(statePath, map[string]any{"controller_pid": os.Getpid(), "child_pid": cmd.Process.Pid, "lane": lane, "experiment": plan.Name, "log": logPath, "started": started.Format(time.RFC3339), "state": "running"})
			done := make(chan error, 1)
			go func() { done <- cmd.Wait() }()
			ticker := time.NewTicker(60 * time.Second)
		waitLoop:
			for {
				select {
				case err = <-done:
					break waitLoop
				case <-ticker.C:
					exec.Command(python, "arena_resources.py").Run()
				}
			}
			ticker.Stop()
		}
		log.Close()
		jobCancel()
		if err != nil {
			failure := map[string]any{"name": plan.Name, "error": err.Error(), "elapsed_seconds": time.Since(started).Seconds(), "log": logPath}
			writeJSON(filepath.Join("competition/experiments", plan.Name, "failure.json"), failure)
			journal("failure", plan.Name+": "+err.Error())
		} else {
			journal("scheduler", fmt.Sprintf("%s 完成，耗时 %.1f 秒。", plan.Name, time.Since(started).Seconds()))
		}
		exec.Command(python, "arena_report.py").Run()
	}
	writeJSON(statePath, map[string]any{"controller_pid": os.Getpid(), "lane": lane, "state": "finished", "finished": time.Now().Format(time.RFC3339)})
	return attempted
}

func main() {
	python := flag.String("python", "/opt/anaconda3/bin/python3", "Python executable")
	maximum := flag.Int("max-experiments", 30, "Maximum new experiments per CPU/GPU lane")
	hours := flag.Float64("hours", 4.5, "Wall-clock experiment budget, preserving finalization separately")
	lane := flag.String("lane", "all", "all, cpu or gpu; all overlaps CPU and MPS experiments")
	finalize := flag.Bool("finalize", true, "Run arena_finalize.py after candidate queue/budget finishes")
	flag.Parse()
	if *lane != "all" && *lane != "cpu" && *lane != "gpu" {
		panic("lane must be all, cpu or gpu")
	}
	if _, err := os.Stat("INSTRUCTIONS.md"); err != nil {
		panic("Run from the TechArena2026 project root")
	}
	os.MkdirAll("competition/logs", 0755)
	ctx, cancel := context.WithTimeout(context.Background(), time.Duration(*hours*float64(time.Hour)))
	defer cancel()
	awake := exec.Command("caffeinate", "-dimsu", "-w", strconv.Itoa(os.Getpid()))
	if err := awake.Start(); err == nil {
		defer awake.Process.Kill()
	}
	journal("scheduler", fmt.Sprintf("Go 调度启动 PID=%d，通道 %s，每通道最大新实验 %d，预算 %.2f 小时。", os.Getpid(), *lane, *maximum, *hours))
	env := append(os.Environ(), "OMP_NUM_THREADS=6", "OPENBLAS_NUM_THREADS=1", "VECLIB_MAXIMUM_THREADS=1", "PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.55", "PYTORCH_MPS_LOW_WATERMARK_RATIO=0.4")
	used := [2]int{}
	// A completed GPU representation may unlock a CPU head, which in turn
	// unlocks a GPU scorer. Drain newly eligible work without exceeding either
	// lane's total experiment cap or the shared wall-clock budget.
	for round := 0; round < 5 && ctx.Err() == nil; round++ {
		before := used[0] + used[1]
		var group sync.WaitGroup
		for index, name := range []string{"cpu", "gpu"} {
			if (*lane == "all" || *lane == name) && used[index] < *maximum {
				group.Add(1)
				go func(index int, name string) {
					defer group.Done()
					used[index] += runLane(ctx, *python, name, *maximum-used[index], env)
				}(index, name)
			}
		}
		group.Wait()
		if used[0]+used[1] == before {
			break
		}
	}
	finalFailed := false
	if *finalize {
		finalLock, err := lockLane(context.Background(), "competition/.finalize.lock")
		if err != nil {
			panic("Cannot acquire finalization lock: " + err.Error())
		}
		defer finalLock.Close()
		cmd := exec.Command(*python, "-u", "arena_finalize.py")
		cmd.Env = env
		cmd.Stdout = os.Stdout
		cmd.Stderr = os.Stderr
		if err := cmd.Run(); err != nil {
			journal("failure", "最终导出失败："+err.Error())
			finalFailed = true
		}
		if !finalFailed {
			verify := exec.Command(*python, "-u", "arena_verify.py")
			verify.Env = env
			log, err := os.Create("competition/logs/final_acceptance.log")
			if err != nil {
				journal("failure", "无法创建验收日志："+err.Error())
				finalFailed = true
			} else {
				verify.Stdout = log
				verify.Stderr = log
				if err := verify.Run(); err != nil {
					journal("failure", "独立验收失败，详见 competition/logs/final_acceptance.log："+err.Error())
					finalFailed = true
				}
				log.Close()
			}
		}
	}
	state := "finished"
	if finalFailed {
		state = "failed"
	}
	writeJSON("competition/controller.json", map[string]any{"controller_pid": os.Getpid(), "state": state, "finished": time.Now().Format(time.RFC3339)})
	journal("scheduler", "本轮 Go 调度结束；所有日志、配置、检索依据和指标已保存。")
	exec.Command(*python, "arena_report.py").Run()
	if finalFailed {
		if awake.Process != nil {
			awake.Process.Kill()
		}
		os.Exit(1)
	}
}
