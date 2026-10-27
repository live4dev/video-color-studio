# Freedive Color Studio

Локальное desktop-приложение для консервативной цветокоррекции подводных фото и видео.
Windows/macOS, русский интерфейс, Natural / Cinematic / Vivid. Исходники не изменяются;
лица, тело, предметы, текстура и кадрирование не пересоздаются. Облако и API-ключи не нужны.

Реализованы фото-MVP, локальные маски и серии, потоковая обработка видео. Фактические
результаты этапов и статус проверок каждой платформы — в [MILESTONES.md](MILESTONES.md)
и [TESTING.md](TESTING.md). Наличие сборочного сценария не означает запуск на целевой ОС.

## Работа в приложении

1. Откройте файлы или папку кнопкой «Импорт», через меню или перетаскиванием.
   Фото: JPEG, PNG, TIFF. Видео: SDR MP4/MOV с H.264 или HEVC.
2. Выберите стиль. Сравните до/после: линия, отдельные виды, масштаб, панорамирование,
   fit и настоящий 100% с загрузкой исходных пикселей, включая Retina.
3. Настройте свет, баланс, красный канал, прозрачность и детали. Во вкладке
   «Области и маски» проверьте кожу, человека и воду. При необходимости используйте
   кисть добавить/вычесть, рамку человека и линию вода/воздух. Воздух имеет собственную
   экспозицию. При низкой уверенности автоматическая кожа остаётся без коррекции.
4. Выберите эталон фотографии для согласования палитры. Переносятся ограниченные
   статистики кожи/воды; освещение, структура, лицо и одежда не копируются.
5. Для видео выберите начало и длину превью до 15 секунд. Готовое превью воспроизводится
   в окне как сравнение без звука; кнопка «Открыть со звуком» запускает локальный плеер.
   Исходные звуковые дорожки сохраняются в самом файле.
6. Экспортируйте выбранный файл или серию. Отметки слева исключают файлы из серии;
   Ctrl/Cmd-клик и Ctrl/Cmd+A выбирают группу. «Настройки → выбранным» переносит
   регуляторы, сохраняя индивидуальные штрихи. Файлы обрабатываются последовательно.
7. «Файл → Сохранить проект» сохраняет `.fcsproj`: пути, включение, настройки,
   маски, эталон, анализ и папки. Пользовательский JSON-пресет переносит регуляторы.

Каждый исходник создаёт отдельный файл и `<имя>.<ext>.freedive.json` с параметрами,
анализом, версиями и контрольными суммами. Существующие результаты не перезаписываются.
Отмена удаляет незавершённый текущий результат и сохраняет завершённые файлы серии.

Фото: JPEG с качеством 1–100, PNG/TIFF в 8 или 16 бит, исходный размер и sRGB.
Видео: H.264, SDR Rec.709, исходное разрешение и временные метки, включая VFR;
звук переносится без перекодирования. Экспорт проверяет кадры, временные метки
и содержимое звуковых пакетов до публикации готового результата.

## Готовая сборка

Сборки и архивы находятся в `dist/`. На macOS распакуйте ZIP и откройте
**Freedive Color Studio.app**. На Windows откройте приложение из portable-папки.
Python, pip, установленный FFmpeg и терминал конечному пользователю не нужны.
Статус конкретных артефактов и неподтверждённые платформы перечислены в TESTING.md.
Developer ID, Authenticode и notarization без сертификатов не выполняются.

## Разработка

Проверенное окружение: Python 3.12.13, PySide6 Essentials 6.8.3, NumPy/OpenCV,
Pillow/tifffile, PyAV 15.1.0 с локальными библиотеками FFmpeg. Версии и хеши —
в `uv.lock`. Сеть нужна для подготовки окружения, но не для работы приложения.
PPHumanSeg Apache-2.0 уже находится в `src/freedive_color/assets/models/`.

macOS Apple Silicon (target 13 исключает случайный выбор wheel для macOS 14):

```sh
uv sync --frozen --python-platform aarch64-apple-darwin --extra dev --extra build
uv run --no-sync python -m freedive_color
```

На Intel Mac target — `x86_64-apple-darwin`. На Windows:

```powershell
uv sync --frozen --extra dev --extra build
uv run --no-sync python -m freedive_color
```

На macOS для совместной работы PyAV и OpenCV используется сборка OpenCV без
FFmpeg/AVFoundation: [docs/NATIVE_OPENCV.md](docs/NATIVE_OPENCV.md). Это исключает
дублирование нативных классов видео. Готовый дистрибутив включает нужные библиотеки.
После установки такого wheel используйте `uv run --no-sync`, чтобы не заменить его
вариантом из реестра. Можно сразу открыть файл: `python -m freedive_color ./dive.mp4`.

## Командная строка

CLI и GUI используют общий движок. Примеры из настроенного окружения:

```sh
python -m freedive_color --input photo.jpg --output output/natural.png --preset natural
python -m freedive_color --input photo.tiff --output output/photo16.png --bit-depth 16
python -m freedive_color --input photo.jpg --output output/local.png --skin-warmth 0.2 --water-hue -0.1
python -m freedive_color --input input --output output --preset cinematic
python -m freedive_color --project dive.fcsproj --output output
python -m freedive_color --input dive.mp4 --output output/dive.mp4 --preset natural
python -m freedive_color --input dive.mp4 --output output/preview.mp4 --preview-start 12 --preview-seconds 10
python -m freedive_color --input dive.mp4 --output output/repeat.mp4 --recipe output/dive.mp4.freedive.json
```

`--settings preset.json` переносит регуляторы; `--set exposure=0.15` меняет отдельный
общий параметр; `--reference photo.jpg` задаёт эталон; `--global-only` отключает
локальные операции. `--recipe` проверяет SHA-256 и повторяет сохранённый анализ,
маски, эталон и диапазон видеопревью. Используйте новое имя результата при повторе.
`--temp-directory` выбирает место временной обработки. JSON не исполняет код.

## Проверки и сборка

```sh
uv run --no-sync ruff check .
uv run --no-sync ruff format --check .
uv run --no-sync pytest -q --junitxml=artifacts/tests.xml
uv run --no-sync python tools/generate_fixtures.py
uv run --no-sync python tools/render_local_evidence.py --output artifacts/new-photo-check
uv run --no-sync python tools/generate_video_fixtures.py --output artifacts/new-video-inputs
uv run --no-sync python tools/render_video_evidence.py --inputs artifacts/new-video-inputs --output artifacts/new-video-check
uv run --no-sync python tools/build_desktop.py
uv run --no-sync python tools/smoke_bundle.py
```

На целевой ОС доступны `packaging/macos/build.sh` и `packaging/windows/build.ps1`.
PyInstaller onedir сохраняет заменяемые динамические библиотеки. Аудит macOS
проверяет minimum OS всех Mach-O. Нативное демо открывает настоящее окно; тесты
используют Qt offscreen. Проверка готового bundle очищает окружение Python и,
на macOS при наличии системной утилиты, запрещает сетевой доступ процессу.

## Практические ограничения

- Потерянный красный канал и точный исходный цвет восстановить по одному кадру
  невозможно. Приложение умеренно усиливает существующий сигнал и предупреждает
  о нехватке данных. Морская вода сохраняет естественный синий/зелёный цвет.
- Модель общего назначения может ошибаться на необычных позах, взвеси и экипировке.
  Проверяйте наложение маски. Рамка и кисть позволяют уточнить результат.
- Автомаски видео следуют за движением и сбрасываются при смене шота. Ручные мазки
  и линия воды закреплены в координатах кадра; покадровых ключей для них пока нет.
- Поддерживается SDR. Распознанные PQ/HLG/HDR/Log отклоняются с объяснением;
  Log без идентифицирующих метаданных надёжно определить невозможно.
- Фото до 40 Мп обрабатывается целиком по одному, без плиточного рендера.
  Произвольные ICC в 16 бит, float/multiframe TIFF и часть сжатий не поддерживаются.
  ICC 8 бит преобразуется в sRGB; 16 бит принимает стандартный sRGB с проверкой.
- Выход фото sRGB; видео Rec.709. Управления профилем монитора нет: для точной
  оценки нужен настроенный монитор. Все метаданные не обещаются; GPS — по выбору,
  TIFF 16 бит не переносит EXIF/GPS. Цветовые теги видео записываются явно.
- HEVC/10-bit на выходе, HDR tone mapping, RAW/HEIC и аппаратное кодирование
  не входят в эту версию. Нечётные размеры видео отклоняются без скрытого кропа.
- Демо не доказывает качество на всех глубинах и оттенках кожи. Проверки на чистых
  Windows/macOS и старых версиях ОС отдельно отмечаются в отчёте.

Архитектура и решения — [DECISIONS.md](DECISIONS.md), происхождение моделей,
медиаданных и библиотек — [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md).
