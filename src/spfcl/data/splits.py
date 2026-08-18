from __future__ import annotations

"""Registered train/scientific-test split for the BOLD Moments pilot."""


try:  # pragma: no cover - full remote environment only
    import pandas as pd
    from neuralset.events.transforms import EventsTransform

    class FixedBmdTaskSplit(EventsTransform):
        """Keep the official train/test stimulus boundary unchanged.

        BOLD Moments randomizes the training-stimulus run assignment across
        subjects, so a run-number validation split leaks stimuli between subjects.
        Phase 1 therefore uses a fixed epoch budget with no validation selection:
        all official training timelines remain ``train`` and the disjoint repeated
        test task becomes ``clean_test`` for post-stage evaluation only.
        """

        def _run(self, events: pd.DataFrame) -> pd.DataFrame:
            if "split" not in events:
                raise ValueError("BOLD Moments events are missing the original task/split column")
            output = events.copy(deep=True)
            output["source_task"] = output["split"].astype(str)
            allowed = output.source_task.isin(["train", "test"])
            output = output.loc[allowed].copy()
            output.loc[output.source_task == "train", "split"] = "train"
            output.loc[output.source_task == "test", "split"] = "clean_test"
            required = {"train", "clean_test"}
            if not required.issubset(set(output.split)):
                raise ValueError(
                    "FixedBmdTaskSplit requires official training and test timelines"
                )
            return output

except ImportError:
    FixedBmdTaskSplit = None  # type: ignore[assignment]
