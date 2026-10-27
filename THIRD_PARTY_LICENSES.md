# Зависимости и происхождение

## Модель сегментации этапа B

`src/freedive_color/assets/models/human_segmentation_pphumanseg_2023mar.onnx`:
OpenCV Zoo / PaddleSeg PPHumanSeg, Apache-2.0, 6 163 938 байт.
SHA-256 `552d8a984054e59b5d773d24b9b12022b22046ceb2bbc4c9aaeaceb36a9ddf24`.
Лицензия, источник и описание находятся рядом с моделью. Модель только классифицирует
пиксели; не генерирует изображение, не обращается в сеть и работает на CPU через
OpenCV DNN. Уточнение GrabCut выполняется локально с фиксированными параметрами.

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
| Pillow / LittleCMS | EXIF, ICC, декодирование | MIT-CMU / MIT; https://pillow.readthedocs.io/en/stable/about.html#license |
| tifffile | 16-bit TIFF | BSD-3-Clause; https://github.com/cgohlke/tifffile |
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

**FFmpeg не поставляется и не вызывается в этапе A.** OpenCV wheel может содержать
свои media-библиотеки и LGPL-компоненты; их notices собираются из wheel. Это не
настроенный backend видео. В этапе C нужна отдельная проверка конкретных бинарников
FFmpeg/ffprobe, доступных кодеков и условий распространения. Произвольная GPL-сборка
не должна автоматически попасть в релиз.

Тестовое фото, автор и public-domain dedication описаны в
`tests/fixtures/ATTRIBUTION.md`. В экспортируемые фотографии надписи не добавляются.
Сборки MVP не имеют Developer ID/Authenticode и notarization; автоматическая
ad-hoc подпись Mach-O при freeze не равнозначна подписи издателя.
