# autoCCT
This script makes CCT matrix using x-rite or spydercheckr 24 colorcheckers to correct camera colors.
To install requirements, use the command:
  
  **pip install -r requirements.txt**
  
# Инструкция
Перед началом калибровки убедитесь, что вы снимаете мишень в необходимых условиях баланса белого.

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

Br: 0  Bg: 0  Bb: 0

    
# Instructions
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

Br: 0 BG: 0 BB: 0
