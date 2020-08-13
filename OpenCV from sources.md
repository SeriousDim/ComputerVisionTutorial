# Установка OpenCV из исходников

**Для чего?** Чтобы иметь весь функционал OpenCV, например, функции SIFT. *opencv-python* из pip не предоставляет такой функционал.
**Примечание:** SURF все еще недоступен, но SIFT работает

## Статьи

1. [Ссылка](https://docs.opencv.org/master/d3/d52/tutorial_windows_install.html). Смотреть **второе** видео. Нужно установить только **Cmake**. Исходники можно скачать с официального сайта [OpenCV](https://opencv.org/releases/). 

2. [Ссылка](https://cv-tricks.com/how-to/installation-of-opencv-4-1-0-in-windows-10-from-source/). Здесь расписаны некоторые галочки (флаги) при сборке в Cmake.

## Установка и сборка исходников (по второй статье)

1. Устанавливаем Python
2. Устанвливаем Visual Studio и плагины для него: приложения для C++ и Python
3. Скачиваем исходники c офиц. сайта OpenCV и распаковываем их
4. Скачиваем OpenCV-contrib [здесь](https://github.com/opencv/opencv_contrib.git). Ссылка есть в второй статье.
5. Распаковываем OpenCV-contrib в ту же папку, что и OpenCV, для удобства. Например: *C:/opencv/*
6. Открываем Cmake
7. Добавляем пути и нажимаем Configure. Делаем все так, как во второй статье
8. Если нужно, выбираем флаги и СНОВА нажимаем Configure
9. Нажимаем Generate
10. В папке в генерированным содержимым откроем OpenCV.snl
11. Выбираем в VS вверху Release и x64
12. Сбилдим файл CMakeTargets/ALL_BUILD
13. Сбилдим файл CMakeTagrets/INSTALL

## Сборка для Python

1. Скачаем через pip opencv-python и opencv-contrib-python
2. Проверяем все ли рабоатет: вводим import cv2
