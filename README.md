# autoCCT
This script makes CCT matrix using x-rite or spydercheckr 24 colorcheckers to correct camera colors.
To install requirements, use the command:
  
  **pip install -r requirements.txt**
  
Connect and authorise phone to adb to automatically get new photos or use **--nophone** to get colorchecker photo locally (save it to last_photo.jpg in the folder from you're starting script)

Normally with adb connected phone you should use script like that:

  **python autoCubes.py --from0 --matrix**
 
 or
  
  **python autoCubes.py --from0 --gcam --matrix**
  for gcam
  
After that the script will save customCCT.txt to PhotonCamera folder and will save it locally to customCCT_autoCubes.txt
There's also different keys, that i'll explain later.

# Русский
Этот скрипт делает CCT матрицу с помощью цветовой мишени spydercheckr 24 или аналогичной мишени x-rite.
Для начала проверьте, установлен ли у вас python версии не ниже 3.6 с помощью команды

  **python -V**

Затем установите необходимые библиотеки с помощью команды

  **pip install -r requirements.txt**

Подключите телефон через adb (приложено) и авторизуйте его, если это не было сделано ранее, чтобы автоматически загружать фото из телефона или используйте ключ **--nophone** чтобы использовать локальное фото. По умолчанию локальное фото ищется в папке, откуда запущен скрипт(!не где лежит, а откуда запущен!) и по названию last_photo.jpg
Если вы хотите указать другой файл, можно отправить имя файла скрипту, например **python autoCubes.py --from0 --nophone some_photo.jpg**

В обычном случае с телефоном, подключенным по adb, используйте скрипт так:

  **python autoCubes.py --from0 --matrix --gcam**
  
для gcam
  
или

  **python autoCubes.py --from0 --matrixes**
  
  для PhotonCamera
  
Если вы хотите использовать цветовую мишень x-rite, добавьте ключ --xrite

  **python autoCubes.py --from0 --matrix --gcam --xrite**
 
  Если калибруете две матрицы для использования в PhotonCamera, по умолчанию калибруется теплая матрица(для теплого света). Если вы хотите калибровать холодную матрицу, добавьте ключ **--cool**
  
  Если вы не уверены в результатах, можно проверить, хорошо ли считались точки с эталона с помощью ключа **--showpoints**
По окончании калибровки скрипт предложит вам уточнить калибровку. Вы можете сначала сделать несколько фото и проверить, требуется ли это.
