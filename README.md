# AgeingSignDetect-beta-
A model to classify different signs of ageing such as puffy eyes, wrinkles, dark spots etc. on the face. Localizes the puffy eyes area if they are present.<br>

Employs the EfficientNet model. <br>

##Installation<br>

>pip install -r requirements.txt<br>
>python AgeingSignDetect.py --help<br>
>python AgeingSignDetect.py --model ./model.json --weight ./weights.h5 --cascade ./haarcascade_frontalface_default.xml --shape-predictor ./shape_predictor_81_face_landmarks.dat
 --image <path_to_image_file>
