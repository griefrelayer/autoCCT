# autoCCT
This script makes CCT matrix using x-rite or spydercheckr 24 colorcheckers to correct camera colors.
  
# Инструкция

Чтобы использовать этот скрипт, вам понадобятся **python** не менее версии 3.6.0 и **pip**

Перейдите по ссылке https://www.python.org/downloads/  чтобы скачать и установить Python

Далее используйте командную строку, чтобы установить pip:

  **curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py**

  **python get-pip.py**

Установите зависимости:
  
  **pip install -r requirements.txt**

Запускайте скрипт командой:
  
  **python autoCubes.py**
  
Добавляйте ключи через пробел после, при необходимости. Например:

  **python autoCubes.py --gcam**
  
  или
  
  **python autoCubes.py --nophone --debug --showpoints --nowb**
  
  
Перед началом калибровки убедитесь, что вы снимаете мишень в необходимых условиях баланса белого.

Подключите телефон через adb (приложено) и авторизуйте его, если это не было сделано ранее, чтобы автоматически загружать фото из телефона. 

Или используйте ключ **--nophone** чтобы использовать локальное фото. По умолчанию локальное фото ищется в папке, откуда запущен скрипт(!не где лежит, а откуда запущен!) и по названию last_photo.jpg

Если вы хотите указать другой файл, можно отправить имя файла скрипту, например 
  
  **python autoCubes.py --nophone some_photo.jpg**

(Если вы калибруете по фото без телефона, то не получится использовать уточнение матрицы, потому что для этого нужно сделать снимок с рассчетной матрицей)

Если вы будете использовать одну матрицу(gcam), снимайте в условиях нейтрального освещения
(дневной свет, лампа дневного света)

Снимайте цветовую мишень так, чтобы белый квадратик мишени был слева.

При этом не важно, горизонтально или вертикально.

Для Gcam добавьте ключ **--gcam**

Для использования с чекером x-rite  добавьте ключ **--xrite**

Если калибруете две матрицы/кубы для теплой или холодной температур, по умолчанию калибруется теплая матрица.

Для калибровки холодной добавьте ключ **--cool**

Для двух матриц добавьте **--matrixes**

(правильно matrices, но у некоторых талантливых разрабов проблемы с английским)

Для куба добавьте **--cube**

Для кубов **--cubes**


Если вы снимаете на PhotonCamera и телефон подключен по adb, 
то при калибровке автоматически выставится матрица для калибровки.

Если вы снимаете на Gcam с функцией cct, 
перед калибровкой отключите настройки цвета и выставьте матрицу следующим образом:

Rr: 1  Rg: 0  Rb: 0

Gr: 0  Gg: 1  Gb: 0

Br: 0  Bg: 0  Bb: 1

    
# Instructions

To use this script you will need working **python > 3.6.0** and **pip** installed
https://www.python.org/downloads/  to download and install python (latest will do)
then use command line:
**curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py**
**python get-pip.py**

To install requirements, use the command:
  
  **pip install -r requirements.txt**
  
  
Before you start calibration, make sure that you shoot the color checker at the necessary white balance conditions.
If you use one matrix (GCAM), remove in neutral lighting conditions
(daylight, daylight lamp)

Remove the color target so that the white color checker square is on the left.

It does not matter, horizontally or vertically.

For GCAM, add the key **--gcam**

To use with X-Rite Checker, add the **--xrite** key.

If you calibrate two matrices / cubes for warm or cold temperatures, a warm matrix is calibrated by default.

For Cold Calibration Add key **--cool**

For two matrices add **--matrixes**

For cube add **--cube**

For cubes **--cubes**

If you shoot on PhotonCamera and the phone is connected to ADB,

The calibration matrix is automatically adjusting during calibration.

If you are shooting on GCAM with a CCT function,

Turn off the color settings before calibrating and set the matrix in settings as follows:

RR: 1 RG: 0 RB: 0

GR: 0 GG: 1 GB: 0

Br: 0 BG: 0 BB: 1
