# Freedive Color Studio

**Обновление этапа B:** раздельные локальные маски кожи/человека/воды, кисти,
ручная граница воды/воздуха, эталон палитры, проекты и последовательный экспорт
серий реализованы. Меню «Файл» открывает и сохраняет `.fcsproj`; отметки в
библиотеке исключают файлы из экспорта. Ctrl/Cmd-клик и Ctrl/Cmd+A выбирают группу,
«Настройки → выбранным» переносит регуляторы, сохраняя индивидуальные штрихи.
Во вкладке «Области и маски» можно проверить маски и поправить их.
Полный отчёт B добавляется в `MILESTONES.md`; описание A ниже сохраняется как история.

```sh
.venv/bin/python -m freedive_color --input ./input --output ./output --preset natural
.venv/bin/python -m freedive_color --project ./dive.fcsproj --output ./output
```

Модель PPHumanSeg (Apache-2.0) поставляется локально, CPU/OpenCV DNN; весов из сети
при запуске не загружается. Уверенность и ограничения показаны в интерфейсе.
Сомнительная автоматическая кожа не корректируется; рамка/кисть задают явную область.
Исходники, размеры, лица и предметы не пересоздаются.

Локальный desktop-фоторедактор для подводных съёмок. **v0.1, этап A: рабочий фото-MVP.**
Запросы в облако, API-ключи, генерация, замена объектов и изменение формы человека отсутствуют.

## Возможности

- Импорт JPEG/JPG, PNG и однокадрового TIFF, файлы и папки, drag & drop.
- Русский интерфейс: библиотека с ленивыми миниатюрами, viewer, настройки, прогресс и отмена.
- Natural, Cinematic и Vivid. Экспозиция, температура, tint, красный канал, контраст,
  света/тени, насыщенность, прозрачность, цветной шум и мягкая резкость.
- До/после/вертикальное сравнение, перетаскивание линии, zoom/pan, fit и настоящий
  100% с загрузкой полного разрешения; масштаб учитывает Retina.
- JPEG с качеством 1–100, PNG/TIFF в 8 или 16 бит. Полный исходный размер без кропа.
- float32; EXIF Orientation; ICC → sRGB через LittleCMS для 8-битных файлов.
- Сохранение/загрузка JSON пресета; индивидуальные настройки импортированных снимков
  живут в текущей сессии. Каждый экспорт получает `<имя>.<ext>.freedive.json` с анализом,
  параметрами, размерами, версиями и SHA-256 оригинала/результата.
- Экспорт одной фотографии в фоне; защита оригинала и существующих результатов,
  проверка свободного места, удаление временных файлов при отмене/ошибке.

## Запуск готовой сборки

На этом компьютере сборки находятся в `dist/`: `.app` и ZIP для Apple Silicon.
Распакуйте ZIP и откройте **Freedive Color Studio.app**. Python и терминал не нужны.
Это сборка для проверки MVP без Developer ID/notarization. Статус реальных проверок,
размеры сборок и превью — в [TESTING.md](TESTING.md).

Для Windows x64 готовятся portable-папка с `.exe` и ZIP. Проверка на Windows и
чистой машине не выполнена в текущей сессии. CI-матрица также содержит macOS arm64/Intel;
наличие workflow не означает, что удалённые сборки уже запущены.

## Разработка и запуск из исходников

Проверенный Python **3.12.13**, зависимости с хешами в `uv.lock`. Для настройки окружения
нужен интернет; само приложение работает локально. Используется `PySide6-Essentials`
(импорт `PySide6`) без WebEngine/Addons. Runtime и notices входят в standalone.

macOS Apple Silicon (задан target 13, чтобы NumPy не выбрал вариант для macOS 14):

```sh
uv sync --frozen --python-platform aarch64-apple-darwin --extra dev --extra build
uv run --no-sync python -m freedive_color
```

Для Intel Mac замените target на `x86_64-apple-darwin`. Windows:

```powershell
uv sync --frozen --extra dev --extra build
uv run --no-sync python -m freedive_color
```

Можно сразу открыть снимок: `uv run --no-sync python -m freedive_color ./photo.jpg`.
Из уже созданного окружения: `.venv/bin/python -m freedive_color` на macOS,
`.venv\Scripts\python.exe -m freedive_color` на Windows.

## Командная строка

CLI не импортирует Qt и использует тот же pipeline, что GUI:

```sh
uv run --no-sync python -m freedive_color --input photo.jpg --output output/photo_natural.png --preset natural
uv run --no-sync python -m freedive_color --input photo.jpg --output output/photo_cinema.jpg --preset cinematic --set exposure=0.15 --quality 95
uv run --no-sync python -m freedive_color --input photo.tiff --output output/photo_16.png --bit-depth 16
uv run --no-sync python -m freedive_color --input photo.jpg --output output/repeat.png --recipe output/photo_natural.png.freedive.json
```

`--settings preset.json` переносит регуляторы на другой снимок; `--recipe sidecar.json`
повторяет также сохранённый анализ и проверяет SHA-256 исходника. Формат и разрядность
выхода задаются отдельно. JSON с параметрами не исполняет код. Существующие файлы
не перезаписываются: для повторной проверки используйте новую папку/имя.

## Тесты и воспроизводимое демо

```sh
uv run --no-sync ruff check .
uv run --no-sync ruff format --check .
uv run --no-sync pytest -q --junitxml=artifacts/tests.xml
uv run --no-sync python tools/generate_fixtures.py
uv run --no-sync python tools/render_evidence.py --output artifacts/new-demo
uv run --no-sync python -m freedive_color --smoke-test tests/fixtures/freediver_moalboal.jpg --evidence artifacts/new-gui-demo
```

Последняя команда открывает настоящее окно, импортирует фото, меняет экспозицию,
экспортирует PNG и сохраняет снимок окна/отчёт. `tests/conftest.py` использует Qt offscreen
только для автоматических тестов. Шорткаты: Cmd/Ctrl+O — импорт, Cmd/Ctrl+Shift+S —
экспорт, F — fit, Cmd/Ctrl+1 — 100%, Alt+1/2/3 — до/сравнение/после.
Двойной щелчок по регулятору сбрасывает его к значению выбранного стиля.

## Сборка

На целевой macOS: `sh packaging/macos/build.sh`. На Windows:
`powershell -ExecutionPolicy Bypass -File packaging/windows/build.ps1`.
После правильно настроенного окружения непосредственно:

```sh
uv run --no-sync python tools/build_desktop.py
uv run --no-sync python tools/smoke_bundle.py
```

PyInstaller standalone выбран вместо Nuitka: штатные hooks для Qt/NumPy/OpenCV,
динамические заменяемые библиотеки, без компиляции NumPy-операций в C повторно.
Архив содержит runtime, библиотеки и фактические тексты лицензий. macOS-сборка
проверяет deployment target всех Mach-O; завышенный minimum OS останавливает выпуск.
На macOS smoke готовой сборки дополнительно запрещает процессу сеть через системный
`sandbox-exec`, если утилита доступна. Это проверка приложения, а не требование к запуску.
Сертификаты издателя и notarization на этом этапе не настроены.

## Ограничения этапа A

- Коррекция глобальная. Семантических масок кожи/человека/воды, кисти, split-level
  редактирования и reference match пока нет. Вода и кожа меняются вместе.
- Импортировать несколько фото/папку можно; экспортируется выбранный снимок.
  Массовый экспорт, палитра серии и проект с восстановлением сессии — этап B.
- Видео, FFmpeg backend, звук и temporal consistency — этап C; их импорт отклоняется.
- Не обещается точный исходный цвет при потерянном красном. Усиливается существующий
  сигнал с лимитом, без безусловного gray-world или переноса зелёного в красный.
- Только SDR sRGB. PQ/HLG, распознанные HDR gain-map/Log, float TIFF и многокадровые
  файлы отклоняются. Log без идентифицирующих метаданных распознать надёжно невозможно.
- 16-bit: RGB/grayscale PNG/TIFF; произвольные ICC в 16 бит не преобразуются. Разрешены
  untagged sRGB (с предупреждением) и собственный стандартный sRGB-профиль экспорта.
  Некоторые варианты TIFF-сжатия требуют отдельного декодера и дадут понятную ошибку.
- Выход всегда sRGB. Нет управления профилем монитора: для точной оценки нужен
  откалиброванный монитор в режиме sRGB. Произвольный wide-gamut экран не проверен.
- Переносятся безопасные EXIF-теги, а не все метаданные/миниатюры/MakerNote/XMP.
  GPS — только по выбору. В TIFF 16 бит остаются базовые TIFF-теги, без EXIF/GPS.
- Лимит входа 40 Мп. Большие файлы обрабатываются по одному; итоговый экспорт
  держит один снимок и промежуточные float-буферы в RAM, это ещё не tile renderer.
- Фото уже обработано автором: демо подтверждает работающий путь, но не качество
  на всём разнообразии глубин, освещения, оттенков кожи и экипировки.

Далее — **этап B**: отдельные маски с оценкой уверенности, ручное уточнение и серии.
Это позволит корректировать кожу, сохраняя воду, одежду и индивидуальный свет.
Архитектура и принятые решения описаны в [DECISIONS.md](DECISIONS.md), права —
в [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md) и `tests/fixtures/ATTRIBUTION.md`.
