## Deteccion de billetes
14/12/2020


Por :   Juan Aguilera ,
        Ricardo Mardones  


####  Referencias: https://github.com/puigalex/deteccion-objetos-video


### Instrucciones:



###### Crear un ambiente virtual con conda y activarlo

```
conda create -n <nombre_venv> python=3.6
conda activate <nombre_venv>
```

###### Instalar PyTorch y torchvision

En caso de que tenga disponible GPU

```
pip install torch===1.5.0 torchvision===0.6.0 -f https://download.pytorch.org/whl/torch_stable.html
```

En caso de que solo se tenga disponible cpu
```
pip install torch==1.5.0+cpu torchvision==0.6.0+cpu -f https://download.pytorch.org/whl/torch_stable.html##
```

###### Instalar los requerimientos para el entorno virtual.

```
pip install -r requirements.txt
```
###### Descargar los pesos necesarios para la deteccion en el siguiente link
```
https://drive.google.com/drive/folders/1WUfFYVQJHfnu9zykEf5jZiSoq6v-qNqy?usp=sharing
```
Donde:

Peso funcional cpu: yolov3_ckpt_97.pth

Peso funcional solo gpu: yolov3_ckpt_6.pth

Agregamos este archivo a la carpeta checkpoints

Notamos que es necesario cambiar este valor en commons.py en las lineas 59 y 65 dependiendo de el peso escogido de la siguente forma:

![Captura de el servidor web](/images_readme/pesos.png)

###### Ejecutar la siguiente linea de codigo para iniciar el webservice

```
python flaskservice.py
```

Esto nos dará el enlace al servidor que nos ayudará a ejecutar la detección de billetes.


![Captura de el servidor web](/images_readme/flaskservice.png)

##### Nos conectamos desde el navegador webpreferido, ya sea en un computador o dispositivo movil y seleccionar un archivo deseado, luego hacemos click en upload para cargarlo al servidor.



# __Disfrutar de una lamentable deteccion de billetes__

![Captura de el servidor web](/images_readme/flaskServiceResult.png)
