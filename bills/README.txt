Deteccion de billetes 03/12/2020
Por:

	Juan Aguilera
	Ricardo Mardones

Referencias:

	https://github.com/puigalex/deteccion-objetos-video

INSTRUCCIONES
Para usar el programa siga los pasos:

-Crear un ambiente virtual con conda y activarlo.
-Ejecutar en consola:

	pip install -r requirements.txt

-Ejecutar la siguiente linea de codigo:
-obs: remplazar <threshold_level> por 0.97 o nivel de confianza deseado.

	python deteccion_video.py --model_def config/yolov3-custom.cfg --checkpoint_model checkpoints/yolov3_ckpt_97.pth --class_path data/custom/classes.names  --weights_path checkpoints/yolov3_ckpt_97.pth  --conf_thres <threshold_level>

-Disfrutar de un pobre reconocimiento de billetes.	


obs: si no ecuentra torch==1.5.0 en los requirements.txt, intente con la siguiente linea:

	pip install torch===1.5.0 torchvision===0.6.0 -f https://download.pytorch.org/whl/torch_stable.html

