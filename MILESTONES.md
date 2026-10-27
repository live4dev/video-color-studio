# Контрольные точки

## A — фото-MVP

Принят предыдущим запуском: 83 теста, реальный JPEG→PNG, нативное окно, macOS
standalone. Подробная историческая проверка находится в TESTING.md.

## B — локальная обработка и серии, 27 октября 2026

**178 passed, 0 failed.** Реализованы CPU person segmentation, отдельная skin mask,
маски воды/подводной части, кисть добавить/вычесть, рамка человека, граница воды,
раздельная экспозиция воздуха, локальная кожа/вода, ограниченный перенос палитры,
сохранение проектов и пользовательских пресетов с локальными параметрами, групповые
настройки, исключения и последовательный экспорт через общий CLI/GUI core.

```sh
.venv/bin/pytest -q --junitxml=artifacts/milestone-b-tests.xml
.venv/bin/python tools/render_local_evidence.py --output artifacts/new-b-check
.venv/bin/python -m freedive_color --input input --output output --preset natural
.venv/bin/python -m freedive_color --project dive.fcsproj --output output
.venv/bin/python -m freedive_color --smoke-test tests/fixtures/freediver_moalboal.jpg --evidence artifacts/new-b-gui --show-mask skin
```

Доказательства: `artifacts/milestone-b-final/evidence.json`, clean outputs и отдельные
QA-наложения в той же папке; нативный снимок окна — `artifacts/milestone-b-gui`.
Серия: **20 completed, 0 failed**, SHA-256 оригиналов не изменены. Воздух: max change
**0.0**. Эталон сохраняет различия света: диапазон яркости серии 0.15106 → 0.15297.
На реальном снимке person coverage 8.908%, skin coverage 0.3914%, confidence 0.88;
костюм, руки и ласта защищены в осмотренном результате. Кожа не включает весь силуэт.

Модель общего назначения: необычные позы, взвесь и сильные цветовые сдвиги могут
потребовать рамки/кисти. Один реальный снимок и цветовые мишени не доказывают
качество на всех оттенках кожи и глубинах. Уверенность и предупреждения доступны.
Нормализованные маски повторяются на полном разрешении; изменившийся исходник
требует явного повторного импорта со сбросом старой разметки.

Следующий этап C: локальный потоковый video backend, шоты и temporal consistency,
proxy/timeline, звук, VFR/PTS и проверка экспорта. Сборка B отдельно не выдаётся
за завершение D: финальные дистрибутивы будут пересобраны после проверки C.
