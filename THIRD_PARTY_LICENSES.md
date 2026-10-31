# Зависимости и происхождение

## Модель сегментации этапа B

`src/freedive_color/assets/models/human_segmentation_pphumanseg_2023mar.onnx`:
OpenCV Zoo / PaddleSeg PPHumanSeg, Apache-2.0, 6 163 938 байт.
SHA-256 `552d8a984054e59b5d773d24b9b12022b22046ceb2bbc4c9aaeaceb36a9ddf24`.
Лицензия, источник и описание находятся рядом с моделью. Модель только классифицирует
пиксели; не генерирует изображение, не обращается в сеть и работает на CPU через
OpenCV DNN. Уточнение GrabCut выполняется локально с фиксированными параметрами.

## Семантическая модель видимой кожи

Google MediaPipe SelfieMulticlass 256×256 float32, версия 1, классифицирует фон,
волосы, кожу тела, кожу лица, одежду и аксессуары. Веса Google LLC распространяются
по Apache-2.0, что отдельно подтверждено официальной карточкой модели:
[Model Card Multiclass Segmentation](https://storage.googleapis.com/mediapipe-assets/Model%20Card%20Multiclass%20Segmentation.pdf).
Официальный TFLite преобразован в ONNX без переобучения или изменения весов;
источники, SHA-256, параметры ввода/вывода и рецепт находятся в
`src/freedive_color/assets/models/SELFIE_MULTICLASS_METADATA.json` и
`tools/convert_skin_model.py`. Apache notice включён рядом с моделью.
Конвертация проверяется сравнением численных выходов TFLite и ONNX на синтетических
данных и открытой тестовой фотографии. TensorFlow, tf2onnx и инструменты конвертации
не входят в зависимости конечного приложения.

ONNX Runtime 1.23.2 (Microsoft, MIT) выполняет локальный CPU inference. Полные
notices ORT и включённых компонентов собираются из фактического wheel; телеметрия
отключена до создания inference session. Модель не обучена специально на подводных
снимках: оценки модели не являются процентом точности; сомнительная автомаска
отключается, а ручная правка остаётся доступной.

Исходный код Freedive Color Studio распространяется по MIT (LICENSE).
Версии и хеши wheel-дистрибутивов фиксирует `uv.lock`; установка только из PyPI
и Python runtime из python-build-standalone через uv. Код моделей не загружается.
Приложение во время работы не обращается в сеть.

| Компонент | Назначение | Основная лицензия / источник |
| --- | --- | --- |
| CPython | Локальный runtime | PSF; https://www.python.org/psf/license/ |
| Qt / PySide6 Essentials / Shiboken6 | Desktop UI, Widgets | LGPL-3.0 / GPL / commercial; https://doc.qt.io/qtforpython-6/licenses.html |
| NumPy | Float32 вычисления | BSD-3-Clause; https://numpy.org/doc/stable/license.html |
| OpenCV headless | Фильтры, 16-bit PNG | Apache-2.0 (OpenCV 4.5+); https://opencv.org/license/ |
| ONNX Runtime 1.23.2 | CPU-сегментация видимой кожи | MIT; https://github.com/microsoft/onnxruntime |
| Pillow / LittleCMS | EXIF, ICC, декодирование | MIT-CMU / MIT; https://pillow.readthedocs.io/en/stable/about.html#license |
| tifffile | 16-bit TIFF | BSD-3-Clause; https://github.com/cgohlke/tifffile |
| PyAV 15.1.0 macOS / 16.1.0 Windows | PTS-aware decode/encode/remux | BSD; https://github.com/PyAV-Org/PyAV |
| FFmpeg 7.1.1 macOS / 8.0.1 Windows, x264/x265 | Локальный media backend | Комбинация GPLv3; точные исходники/patches в отдельном source ZIP |
| PyInstaller | Сборка standalone | GPL-2.0-or-later с исключением для bundling; https://pyinstaller.org/en/stable/license.html |
| pytest / pytest-qt / Ruff | Только разработка | MIT / MIT / MIT |

Выбран только `PySide6-Essentials` (тот же Python namespace `PySide6`): Core/Gui/Widgets
без WebEngine и ненужных Addons. Сборка — PyInstaller onedir, альтернативный путь,
допускаемый ТЗ: быстрый воспроизводимый freeze NumPy/OpenCV/Qt, заменяемые динамические
библиотеки и штатные hooks. Nuitka не нужен для производительности NumPy-операций.
Нет закрытого кода, запрещающего изучение или изменение библиотек.

`tools/collect_licenses.py` переносит фактические notices из установленных wheel,
включая bundled native components, в `build/licenses`. Они входят в `.app`/portable
в папку `licenses/dependencies`; manifest содержит точные версии. Отдельные Qt-модули
могут иметь другие лицензии: в коде используются только Core, Gui и Widgets.

В актуальной macOS-сборке OpenCV собран из проверенного исходника без FFmpeg и
AVFoundation. Его роль — изображения, фильтры, сегментация и optical flow.
PyAV содержит собственные локальные библиотеки FFmpeg; системный FFmpeg не требуется.
Входные demuxer/protocol ограничены локальным MOV/MP4, скрытых сетевых загрузок нет.

**Автономный дистрибутив в целом распространяется на условиях GPLv3.** Собственные
исходные файлы приложения сохраняют лицензию MIT. В PyAV 15.1.0 community wheel
есть x264/x265; patch производителя убирает их из GPL-проверок configure, но не
меняет лицензии. Строка LGPL из avcodec не принимается за описание всей комбинации.
Происхождение: pyav-ffmpeg `7.1.1-6`, FFmpeg `7.1.1`; точные записи находятся в
[packaging/media/NOTICE.md](packaging/media/NOTICE.md). Полные тексты лицензий,
совпадающие исходники библиотек, patches и рецепты сборки предоставлены рядом
с приложением в `Freedive_Color_Studio_Corresponding_Source.zip` и его JSON checksum.
Инструкции — [docs/CORRESPONDING_SOURCE.md](docs/CORRESPONDING_SOURCE.md).

Qt/PySide/Shiboken используются динамически. Их исходники 6.8.3 и полные LGPL/GPL
тексты также входят в source ZIP. Изменять и пересобирать библиотеки разрешено;
ограничений на изучение и отладку изменений нет. На macOS изменённую локальную
сборку можно заново подписать технической ad-hoc подписью `codesign --force --deep
--sign - "Freedive Color Studio.app"`, без сертификата издателя.

Архив источников фиксирует проверенную платформу. Windows wheel содержит другой
набор DLL; его происхождение и совместимость лицензий проверяются отдельно.
Неполный/несовместимый набор не считается готовым для распространения.

Тестовое фото, автор и public-domain dedication описаны в
`tests/fixtures/ATTRIBUTION.md`. В экспортируемые фотографии надписи не добавляются.
Дополнительный движущийся подводный ролик для QA: It is a wonderful world,
[Backstroke Underwater](https://commons.wikimedia.org/wiki/File:Backstroke_Underwater.webm),
CC BY-SA 4.0. Его атрибуция и лицензия относятся к самому медиа и производным
QA-результатам; тестовый ролик не входит в приложение. Контролируемые тестовые
MP4 с изменением света созданы из public-domain фотографии и синтетического звука.
Сборки MVP не имеют Developer ID/Authenticode и notarization; автоматическая
ad-hoc подпись Mach-O при freeze не равнозначна подписи издателя.

## Локальная MCP-интеграция 0.4

Официальные `mcp` и `mcp-types` 2.3.0 — MIT. Версии всей зависимости SDK
закреплены в uv.lock; SDK используется только со stdio. Наличие HTTP/OAuth-модулей
в пакете не означает запуска HTTP-сервера или обращения к облаку. PySide6.QtNetwork
предоставляет локальные сокеты; его условия и исходники относятся к QtBase 6.8.3.

Дополнительные компоненты включают Pydantic/pydantic-core (MIT), rpds-py (MIT),
cryptography (Apache-2.0 OR BSD-3-Clause), cffi (MIT), httpx2/httpcore2 (BSD-3-Clause),
AnyIO, JSON Schema, Starlette/Uvicorn и связанные библиотеки. На Windows требуется
pywin32 312 (PSF/BSD-совместимые условия его точного producer source). Полные
пакетные notices собираются по фактически установленному графу, включая extras.

Точные 28 Python sdists, 140 SHA-проверенных Cargo-архивов и OpenSSL 4.0.3
зафиксированы в `packaging/media/mcp-sources-lock.json`. Версия OpenSSL сверяется
с фактическими macOS FFI и Windows PE/SBOM; Mac native SDK-модули имеют minimum OS 11.
Точная pywin32 source-access запись отдельно отмечена и не выдаётся за включённый
архив. Этот lock дополняет прежние обязательные Qt/FFmpeg/OpenCV source gates,
не заменяя и не ослабляя их. Сводные источники и полные notices обновляются вместе
с standalone-дистрибутивами.
