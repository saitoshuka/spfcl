from __future__ import annotations

"""BOLD Moments session-1 localizer adapter.

Upstream TRIBE v2 only enumerates sessions 2--5. This registered study keeps the
official loader untouched while exposing the five same-subject silent-video
localizer runs needed for Phase-1 scientific probes.
"""

from collections.abc import Iterator
from pathlib import Path
from typing import Any

try:  # pragma: no cover - exercised in the full remote environment
    import pandas as pd
    from neuralset.events import study
    from neuralset.utils import get_bids_filepath, read_bids_events
    from tribev2.studies.lahner2024bold import Lahner2024Bold

    class Lahner2024BoldLocalizer(Lahner2024Bold):
        _info = None
        NUM_SUBJECTS = 4

        def model_post_init(self, context: Any) -> None:
            requested = Path(self.path)
            super().model_post_init(context)
            if self.path.name.lower() != "lahner2024bold":
                candidate = requested / "Lahner2024Bold"
                if candidate.is_dir():
                    self.path = candidate

        def iter_timelines(self) -> Iterator[dict[str, Any]]:
            for subject in range(1, self.NUM_SUBJECTS + 1):
                for run in range(1, 6):
                    yield {
                        "subject": subject,
                        "session": 1,
                        "split": "localizer",
                        "run": run,
                    }

        def _load_timeline_events(self, timeline: dict[str, Any]) -> pd.DataFrame:
            tl = dict(timeline)
            task = tl.pop("split")
            bold, _ = self._get_bold_images(timeline, "MNI152NLin2009cAsym")
            n_volumes = int(bold.shape[-1])
            info = study.SpecialLoader(method=self._load_raw, timeline=timeline).to_json()
            fmri = {
                "filepath": info,
                "type": "Fmri",
                "start": 0.0,
                "frequency": 1.0 / self.TR_FMRI_S,
                "duration": n_volumes * self.TR_FMRI_S,
            }
            events_path = get_bids_filepath(
                root_path=self.path / "download",
                filetype="events",
                data_type="Fmri",
                run_padding="01",
                task=task,
                **tl,
            )
            bids = read_bids_events(events_path)
            # Localizer runs contain 18-second fixation blocks with stim_file=n/a.
            # They remain represented by the continuous fMRI event but must not
            # become Video events (there is intentionally no corresponding file).
            stimulus = bids["stim_file"]
            valid_stimulus = stimulus.notna() & stimulus.astype(str).str.strip().str.lower().ne(
                "n/a"
            )
            bids = bids[(bids.trial_type != "oddball") & valid_stimulus]
            stimulus_root = self.path / "stimuli" / "stimulus_set" / "stimuli"
            videos = []
            for event in bids.to_dict("records"):
                relative = Path(event["stim_file"])
                videos.append(
                    {
                        "type": "Video",
                        "start": event["onset"],
                        "duration": event.get("duration", 0.0),
                        "filepath": str(stimulus_root / relative),
                        "localizer_condition": event.get("trial_type", "unknown"),
                    }
                )
            return pd.concat([pd.DataFrame([fmri]), pd.DataFrame(videos)], ignore_index=True)

except ImportError:
    Lahner2024BoldLocalizer = None  # type: ignore[assignment]
