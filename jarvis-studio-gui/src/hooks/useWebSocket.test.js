import { describe, expect, it } from "vitest";
import {
  isMatchingRecordingAcceptance,
  isMatchingRecordingRejection,
  reduceRecordingState,
  reduceSetupProgress,
  resolveRecordingPending,
} from "./useWebSocket";

describe("reduceRecordingState", () => {
  it("tracks only the dedicated recording lifecycle event", () => {
    const idle = { serverId: null, generation: -1, revision: -1, state: "idle" };
    const recording = reduceRecordingState(idle, "recording_state", {
      server_id: "server-a", generation: 1, revision: 1,
      state: "recording", request_id: "start-1", action: "start",
    });
    expect(recording.state).toBe("recording");
    expect(reduceRecordingState(recording, "status", "idle")).toBe(recording);
    const transcribing = reduceRecordingState(recording, "recording_state", {
      server_id: "server-a", generation: 1, revision: 2,
      state: "transcribing", request_id: "stop-1", action: "stop",
    });
    expect(transcribing.state).toBe("transcribing");
    expect(reduceRecordingState(transcribing, "recording_state", {
      server_id: "server-a", generation: 1, revision: 3, state: "idle",
    }).state).toBe("idle");
  });

  it("clears a stale recording state on disconnect without starting again", () => {
    const active = {
      serverId: "server-a", generation: 2, revision: 4, state: "recording",
    };
    const disconnected = reduceRecordingState(active, "disconnected");
    expect(disconnected.state).toBe("idle");
    expect(disconnected.serverId).toBeNull();
  });

  it("ignores older sessions, revisions, and reconnect snapshots", () => {
    const current = {
      serverId: "server-a", generation: 4, revision: 7, state: "recording",
    };
    expect(reduceRecordingState(current, "recording_state", {
      server_id: "server-a", generation: 3, revision: 99, state: "idle",
    })).toBe(current);
    expect(reduceRecordingState(current, "recording_state", {
      server_id: "server-a", generation: 4, revision: 6, state: "idle",
    })).toBe(current);
    expect(reduceRecordingState(current, "recording_state", {
      server_id: "server-a", generation: 4, revision: 7, state: "idle",
    })).toBe(current);
    expect(reduceRecordingState(current, "status", "idle")).toBe(current);
  });

  it("accepts a new backend process snapshot despite its reset generation", () => {
    const current = {
      serverId: "old-server", generation: 8, revision: 20, state: "recording",
    };
    const next = reduceRecordingState(current, "recording_state", {
      server_id: "new-server", generation: 0, revision: 0, state: "idle",
    });
    expect(next.serverId).toBe("new-server");
    expect(next.state).toBe("idle");
  });

  it("resolves pending actions only on matching acceptance or rejection", () => {
    const pending = { action: "start", requestId: "start-2" };
    const accepted = {
      requestId: "start-2", action: "start", state: "recording",
    };
    expect(isMatchingRecordingAcceptance(pending, accepted)).toBe(true);
    expect(resolveRecordingPending(pending, "recording_state", accepted)).toBeNull();
    expect(resolveRecordingPending(pending, "warning", "unrelated warning")).toBe(pending);
    expect(resolveRecordingPending(pending, "error", "unrelated error")).toBe(pending);
    expect(isMatchingRecordingRejection(pending, {
      request_id: "start-2", action: "start", message: "busy",
    })).toBe(true);
    expect(resolveRecordingPending(pending, "recording_rejected", {
      request_id: "different", action: "start",
    })).toBe(pending);
    expect(resolveRecordingPending(pending, "recording_rejected", {
      request_id: "start-2", action: "start",
    })).toBeNull();

    const stop = { action: "stop", requestId: "stop-1" };
    expect(resolveRecordingPending(stop, "recording_state", {
      requestId: "stop-1", action: "stop", state: "transcribing",
    })).toBeNull();
    const cancel = { action: "cancel", requestId: "cancel-1" };
    expect(resolveRecordingPending(cancel, "recording_state", {
      requestId: "cancel-1", action: "cancel", state: "idle",
    })).toBeNull();
    expect(resolveRecordingPending(cancel, "disconnected")).toBeNull();
  });
});

describe("reduceSetupProgress", () => {
  it("tracks each asset once, in arrival order, through to complete", () => {
    let s = reduceSetupProgress(null, { stage: "start", total: 2 });
    s = reduceSetupProgress(s, { stage: "downloading", asset: "whisper", label: "Whisper" });
    s = reduceSetupProgress(s, { stage: "downloading", asset: "piper" });
    s = reduceSetupProgress(s, { stage: "done", asset: "whisper", label: "Whisper" });
    s = reduceSetupProgress(s, { stage: "error", asset: "piper", error: "timeout" });
    expect(s.order).toEqual(["whisper", "piper"]);
    expect(s.items.whisper.status).toBe("done");
    expect(s.items.piper).toEqual({ label: "piper", status: "error", error: "timeout" });
    expect(reduceSetupProgress(s, { stage: "complete" }).complete).toBe(true);
  });

  it("adopts a snapshot wholesale (late-connecting HUD)", () => {
    const s = reduceSetupProgress({ total: 9, order: ["x"], items: {}, complete: false }, {
      stage: "snapshot", order: ["a"], items: { a: { status: "done" } }, complete: true,
    });
    expect(s).toEqual({ total: 1, order: ["a"], items: { a: { status: "done" } }, complete: true });
  });

  it("ignores junk and never mutates the previous state", () => {
    const prev = { total: 1, order: [], items: {}, complete: false };
    expect(reduceSetupProgress(prev, null)).toBe(prev);
    reduceSetupProgress(prev, { stage: "downloading", asset: "a" });
    expect(prev.order).toEqual([]);
  });
});
