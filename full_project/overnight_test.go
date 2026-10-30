package main

import (
	"context"
	"errors"
	"path/filepath"
	"testing"
	"time"
)

func TestLaneLockSerializesWritersAndHonorsDeadline(t *testing.T) {
	path := filepath.Join(t.TempDir(), "lane.lock")
	first, err := lockLane(context.Background(), path)
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Millisecond)
	defer cancel()
	second, err := lockLane(ctx, path)
	if second != nil {
		second.Close()
		t.Fatal("Two writers acquired the same lane")
	}
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("Expected deadline, got %v", err)
	}
	first.Close()
	third, err := lockLane(context.Background(), path)
	if err != nil {
		t.Fatal(err)
	}
	third.Close()
}
