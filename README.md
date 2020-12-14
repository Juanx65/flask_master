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
###### Ejecutar la siguiente linea de codigo para

```
python flaskservice.py
```

Esto nos dará el enlace al servidor que nos ayudará a ejecutar la detección de billetes.


![Captura de el servidor web](/images_readme/flaskservice.png)

##### Seleccionar un archivo desde el computador (por el momento) y hacer click en upload para cargarlo al servidor.



# __Disfrutar de la pobre detecciond de billetes__

![Captura de el servidor web](/images_readme/flaskServiceresult.png)
