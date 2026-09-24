def run_coordinator():
    import json, os
    import wanferenz.serving.chain as p
    from wanferenz.benchmark.timing import install

    if os.environ.get("WANFERENZ_BENCH_SWEEP_PLAN"):
        from wanferenz.benchmark.latency import attach_latency_checks

        attach_latency_checks(p, json.loads(os.environ["WANFERENZ_BENCH_SWEEP_PLAN"]))
    install(p, json.loads(os.environ.get("WANFERENZ_BENCH_ARMS", "{}")))
    p.run_command()


def run_worker():
    import json, os
    import wanferenz.serving.chain as p
    import wanferenz.serving.partition as stage
    import wanferenz.decoding.advance as fast

    arms = json.loads(os.environ["WANFERENZ_BENCH_STAGE_ARMS"])
    current = {}
    original_recv = p.receive_frame

    def recv(*args, **kwargs):
        message = original_recv(*args, **kwargs)
        if isinstance(message, dict) and message.get("op") == "reset":
            mode = message["job_id"].rsplit("-", 1)[0]
            current.clear()
            current.update(arms[mode])
            print(
                "WANFERENZ_BENCH_STAGE_ARM "
                + json.dumps({"jobId": message["job_id"], **current}),
                flush=True,
            )
        return message

    p.receive_frame = recv
    original_reset = stage.LayerPartition.reset

    def reset(self):
        original_reset(self)
        if current:
            self._snapshot_mode = current["snapshot"]
            fast.V4_DSPARK_FULL_GRAPH = current["fullGraph"]
            stage.V4_HEAD_GRAPH = current.get("headGraph", False)
            if "fp8" in current:
                import wanferenz.kernels.fp8_vector as fg
                import wanferenz.graphs.decoder as wl

                mode = current["fp8"]
                if fg._GEMV_MODE != mode:
                    graphs = getattr(self, "_bench_graph_sets", {})
                    graphs[fg._GEMV_MODE] = self._block_graphs
                    if mode not in graphs:
                        graphs[mode] = [
                            wl.CapturedDecoder(L, self, moe_mode="graph")
                            for L in self.layers
                        ]
                    self._block_graphs = graphs[mode]
                    self._bench_graph_sets = graphs
                    verdicts = getattr(self, "_bench_fp8_verdicts", {})
                    verdicts[fg._GEMV_MODE] = fg._GEMV_VERDICTS
                    fg._GEMV_VERDICTS = verdicts.setdefault(mode, {})
                    self._bench_fp8_verdicts = verdicts
                    fg.V4_FP8_GEMV = mode
                    fg._GEMV_MODE = mode

    stage.LayerPartition.reset = reset
    if any(("fp8" in arm for arm in arms.values())):
        import wanferenz.kernels.fp8_vector as fg

        original_kept = fast._kept_forward_spec

        def kept(self, *args, **kwargs):
            graphs = getattr(self, "_bench_draft_graphs", {})
            self._full_draft_graph = graphs.get(fg._GEMV_MODE)
            outcome = original_kept(self, *args, **kwargs)
            graphs[fg._GEMV_MODE] = self._full_draft_graph
            self._bench_draft_graphs = graphs
            return outcome

        fast._kept_forward_spec = kept
    p.run_command()
