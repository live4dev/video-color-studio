import argparse
import json
from dataclasses import replace
from pathlib import Path

from .core.errors import Cancelled, StudioError
from .core.image_pipeline import export_photo
from .core.presets import PRESETS, get_preset
from .core.project import load_recipe, load_settings


def run(arguments: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Freedive Color Studio — локальная обработка SDR фото"
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--preset", choices=list(PRESETS), default="natural")
    parser.add_argument(
        "--settings", type=Path, help="JSON пресета или sidecar: загрузить регуляторы"
    )
    parser.add_argument(
        "--recipe", type=Path, help="Sidecar: воспроизвести также анализ того же исходника"
    )
    parser.add_argument("--set", action="append", default=[], metavar="PARAMETER=VALUE")
    parser.add_argument("--bit-depth", type=int, choices=(8, 16), default=8)
    parser.add_argument("--quality", type=int, default=95)
    parser.add_argument("--keep-gps", action="store_true")
    args = parser.parse_args(arguments)
    try:
        if args.input.is_dir():
            raise StudioError(
                "В этапе A обрабатывается один снимок. Пакетный экспорт запланирован в B."
            )
        if args.recipe and (args.settings or args.set):
            raise StudioError("Для точного повтора рецепта не добавляйте --settings или --set.")
        analysis = None
        if args.recipe:
            settings, analysis = load_recipe(args.recipe, args.input)
        else:
            settings = load_settings(args.settings) if args.settings else get_preset(args.preset)
            overrides = {}
            for entry in args.set:
                key, value = entry.split("=", 1)
                if key not in settings.to_dict() or key == "preset":
                    raise StudioError(f"Неизвестный числовой параметр: {key}.")
                overrides[key] = float(value)
            if overrides:
                settings = replace(settings, **overrides)
        report = export_photo(
            args.input,
            args.output,
            settings,
            analysis=analysis,
            bit_depth=args.bit_depth,
            quality=args.quality,
            keep_gps=args.keep_gps,
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    except (StudioError, ValueError, OSError) as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False))
        return 130 if isinstance(exc, Cancelled) else 2
    except KeyboardInterrupt:
        print(json.dumps({"error": "Обработка отменена."}, ensure_ascii=False))
        return 130
