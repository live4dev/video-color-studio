"""Headless entry point sharing the complete processing core with the desktop editor."""

import argparse
import json
from dataclasses import replace
from pathlib import Path

from .core.batch import export_batch
from .core.errors import Cancelled, StudioError
from .core.image_pipeline import create_image_reference, export_photo
from .core.local_pipeline import LocalSettings
from .core.media_io import PHOTO_EXTENSIONS
from .core.presets import PRESETS, GradeSettings, get_preset
from .core.project import load_recipe, load_settings, read_json, sha256_file
from .core.reference import PaletteReference
from .core.session import ExportOptions, SessionItem, load_session


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Freedive Color Studio — локальная цветокоррекция")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=Path, help="Фото, MP4/MOV или папка медиа")
    source.add_argument("--project", type=Path, help="Проект .fcsproj / .json")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--preset", choices=list(PRESETS))
    parser.add_argument("--settings", type=Path, help="JSON пресета или sidecar")
    parser.add_argument("--recipe", type=Path, help="Повтор параметров/анализа того же исходника")
    parser.add_argument("--set", action="append", default=[], metavar="PARAMETER=VALUE")
    parser.add_argument("--bit-depth", type=int, choices=(8, 16))
    parser.add_argument("--quality", type=int)
    parser.add_argument("--format", choices=("jpg", "png", "tiff"), help="Формат серии")
    parser.add_argument("--keep-gps", action="store_true")
    parser.add_argument("--reference", type=Path, help="Эталон палитры, без переноса экспозиции")
    parser.add_argument("--global-only", action="store_true", help="Отключить локальные маски")
    parser.add_argument("--mask-edits", type=Path, help="JSON нормализованных штрихов/границы воды")
    parser.add_argument(
        "--preview-seconds", type=float, help="Длительность видео-превью, до 15 секунд"
    )
    parser.add_argument(
        "--preview-start", type=float, default=0, help="Начало участка видео, секунды"
    )
    parser.add_argument("--temp-directory", type=Path, help="Папка временных файлов видео")
    for field in (
        "skin_warmth",
        "skin_strength",
        "water_hue",
        "water_saturation",
        "above_exposure",
    ):
        parser.add_argument("--" + field.replace("_", "-"), type=float)
    return parser


def run(arguments: list[str]) -> int:
    args = build_parser().parse_args(arguments)
    try:
        local_values = {
            key: getattr(args, key)
            for key in (
                "skin_warmth",
                "skin_strength",
                "water_hue",
                "water_saturation",
                "above_exposure",
            )
            if getattr(args, key) is not None
        }
        if args.recipe and (
            args.settings
            or args.set
            or args.project
            or args.preset
            or args.global_only
            or args.reference
            or args.mask_edits
            or local_values
        ):
            raise StudioError("Для повтора рецепта не добавляйте другие параметры обработки.")
        if args.recipe and args.input.is_dir():
            raise StudioError("Рецепт повторяет один исходник, а не папку.")
        if args.global_only and (args.reference or args.mask_edits or local_values):
            raise StudioError(
                "--global-only несовместим с масками, эталоном и локальными параметрами."
            )
        settings = (
            load_settings(args.settings) if args.settings else get_preset(args.preset or "natural")
        )
        overrides = {}
        for entry in args.set:
            key, value = entry.split("=", 1)
            if key not in settings.to_dict() or key == "preset":
                raise StudioError(f"Неизвестный числовой параметр: {key}.")
            overrides[key] = float(value)
        if overrides:
            settings = replace(settings, **overrides)
        preset_data = read_json(args.settings) if args.settings else {}
        preset_local = (
            LocalSettings.from_dict(preset_data["local_settings"])
            if preset_data.get("local_settings")
            else None
        )
        local = (preset_local or LocalSettings()) if not args.global_only else None
        local_values = {
            key: getattr(args, key)
            for key in (
                "skin_warmth",
                "skin_strength",
                "water_hue",
                "water_saturation",
                "above_exposure",
            )
            if getattr(args, key) is not None
        }
        if local_values:
            local = replace(local or LocalSettings(), **local_values)
        edits = None
        if args.mask_edits:
            from .core.masks import MaskEdits

            data = json.loads(args.mask_edits.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                raise StudioError("Правки масок должны быть JSON-объектом.")
            edits = MaskEdits.from_dict(data.get("mask_edits", data))
        reference = create_image_reference(args.reference) if args.reference else None
        analysis, video_recipe = None, None
        if args.recipe:
            data = read_json(args.recipe, max_bytes=32 * 1024 * 1024)
            if data.get("kind") == "video_recipe":
                source_record = data.get("source")
                if args.input.suffix.lower() not in (".mp4", ".mov"):
                    raise StudioError("Рецепт видео требует исходник MP4/MOV.")
                if not isinstance(source_record, dict) or source_record.get(
                    "sha256"
                ) != sha256_file(args.input):
                    raise StudioError("Рецепт создан для другого исходника видео.")
                video_recipe = data
                settings = GradeSettings.from_dict(data.get("settings", {}))
            else:
                settings, analysis = load_recipe(args.recipe, args.input)
            local = data.get("local_settings")
            edits = data.get("mask_edits")
            reference = (
                PaletteReference.from_dict(data["reference"]) if data.get("reference") else None
            )
        if args.project or args.input.is_dir():
            reference_path = None
            if args.preview_seconds is not None or args.preview_start:
                raise StudioError("Превью создаётся для одного видео, а не для серии.")
            if args.recipe:
                raise StudioError("Рецепт повторяет один исходник; для серии используйте проект.")
            if args.project:
                project = load_session(args.project)
                items = project.items
                options = project.export_options
                reference = reference or (
                    PaletteReference.from_dict(project.reference_data)
                    if project.reference_data
                    else None
                )
                reference_path = project.reference_path if reference is None else None
                changed = []
                for item in items:
                    item_settings = settings if args.settings or args.preset else item.settings
                    if overrides:
                        item_settings = replace(item_settings, **overrides)
                    item_local = preset_local or LocalSettings.from_dict(item.local_settings)
                    if local_values:
                        item_local = replace(item_local, **local_values)
                    if args.global_only:
                        item_local = replace(item_local, enabled=False)
                    changed.append(
                        replace(
                            item,
                            settings=item_settings,
                            local_settings=(
                                item_local.to_dict()
                                if local_values or args.global_only or preset_local
                                else item.local_settings
                            ),
                            mask_edits=edits.to_dict() if edits else item.mask_edits,
                        )
                    )
                items = tuple(changed)
                if args.global_only:
                    reference, reference_path = None, None
            else:
                paths = sorted(
                    (
                        path
                        for path in args.input.iterdir()
                        if path.is_file()
                        and path.suffix.lower() in PHOTO_EXTENSIONS | {".mp4", ".mov"}
                    ),
                    key=lambda path: path.name.casefold(),
                )
                items = tuple(
                    SessionItem(
                        path,
                        settings,
                        local_settings=local.to_dict() if local else {},
                        mask_edits=edits.to_dict() if edits else {},
                    )
                    for path in paths
                )
                options = ExportOptions()
            if not items:
                raise StudioError("В папке/проекте нет подходящих файлов.")
            options = replace(
                options,
                suffix="." + args.format if args.format else options.suffix,
                bit_depth=args.bit_depth if args.bit_depth is not None else options.bit_depth,
                quality=args.quality if args.quality is not None else options.quality,
                keep_gps=args.keep_gps or options.keep_gps,
            )
            result = export_batch(
                items,
                args.output,
                options,
                reference=reference,
                reference_path=reference_path,
                temp_directory=args.temp_directory
                or (project.temp_directory if args.project else None),
            )
            print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
            return 130 if result.cancelled else (2 if result.failed else 0)
        if args.input.suffix.lower() in (".mp4", ".mov"):
            from .core.video_pipeline import export_video

            if args.bit_depth not in (None, 8):
                raise StudioError("Видео экспортируется в SDR H.264 8 бит.")
            preview_start, preview_seconds = args.preview_start, args.preview_seconds
            proxy_edge, plan = None, None
            if video_recipe:
                plan = video_recipe.get("plan")
                proxy_edge = video_recipe.get("output", {}).get("proxy_edge")
                if (
                    preview_seconds is None
                    and not preview_start
                    and video_recipe.get("preview", {}).get("enabled")
                ):
                    preview_start = video_recipe["preview"]["requested_start"]
                    preview_seconds = video_recipe["preview"]["requested_seconds"]
            report = export_video(
                args.input,
                args.output,
                settings,
                local_settings=local,
                mask_edits=edits,
                reference=reference,
                preview_start=preview_start,
                preview_seconds=preview_seconds,
                temp_directory=args.temp_directory,
                plan=plan,
                proxy_edge=proxy_edge,
                keep_gps=args.keep_gps,
            )
            print(json.dumps(report, ensure_ascii=False, indent=2))
            return 0
        if args.preview_seconds is not None or args.preview_start:
            raise StudioError("Диапазон превью доступен только для видео.")
        report = export_photo(
            args.input,
            args.output,
            settings,
            analysis=analysis,
            local_settings=local,
            mask_edits=edits,
            reference=reference,
            bit_depth=args.bit_depth if args.bit_depth is not None else 8,
            quality=args.quality if args.quality is not None else 95,
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
