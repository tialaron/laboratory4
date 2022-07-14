import streamlit as st

#import cv2
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
from tensorflow.keras.models import load_model


model_2d = load_model('/app/laboratory4/mnist_2d.h5')    
file_path = '/app/laboratory4/your_file_image.png'

#st.set_page_config(layout="wide")
#st.title("Распознавание рукописных цифр искусственной нейронной сетью (ИНС)")

st.markdown('''<h1 style='text-align: center; color: #F64A46;'
            >Распознавание рукописных цифр искусственной нейронной сетью (ИНС)</h1>''', 
            unsafe_allow_html=True)

img_start = Image.open('/app/laboratory4/pictures/start_picture.png') #
st.image(img_start, use_column_width='auto') #width=450

st.write("""
Лабораторная работа *"Распознавание рукописных цифр искусственной нейронной сетью (ИНС)"* позволяет продемонстрировать 
функционирование реальной нейронной сети, обученной распознавать рукописные цифры.
""")

img_pipeline_mnist = Image.open('/app/laboratory4/pictures/pipeline_for_MNIST_3.png') 
st.image(img_pipeline_mnist, use_column_width='auto', caption='Общая схема лабораторной работы') #width=450

pipe_expander = st.expander("Описание лабораторной работы:")
pipe_expander.markdown(
    """
    \n**Этапы:**
    \n(зелёным обозначены этапы, корректировка которых доступна студенту, красным - этапы, что предобработаны и скорректированы сотрудником лаборатории)
    \n1. База данных MNIST:
    \nБыла использована стандартная база данных MNIST, состоящая из 60000 изображений рукописных цифр размером 28х28 пикселей. [(ссылка на данные)](http://yann.lecun.com/exdb/mnist/);
    \n2. Библиотека слоев:
    \nЗагрузка библиотеки функций-слоев из которых состоит нейронная сеть. [tensorflow](https://www.tensorflow.org/), [numpy](https://numpy.org/doc/stable/reference/index.html);
    \n3. Настройка модели:
    \nПодбираем архитектуру нейронной сети. При этом используется библиотека [matplotlib](https://matplotlib.org/stable/api/index.html) для графического отображения результатов работы модели;
    \n4. Обучение модели:
    \nДанный процесс запускается с помощью встроенных функций библиотеки [tensorflow](https://www.tensorflow.org/);
    \n5. Проверка точности:
    \nТакже происходит с помощью встроенных функций библиотеки [tensorflow](https://www.tensorflow.org/);
    \n6. Функции обработки изображения:
    \nНейронная сеть представляет изображение в виде матрицы чисел. Данные функции преобразуют эту матрицу обратно в изображение, к которому мы привыкли. 
    \nА также позволяют редактировать изображение с целью улучшения распознавания цифры. Библиотека [Python Imaging Library](https://pypi.org/project/Pillow/);
    \n7. Загрузка изображения: 
    \nПроизводится студентами РАНХиГС, непосредственно выполняющими лабораторную работу. Необходимо взять карточку с цифрой, поднести к 
    \nкамере и нажать на кнопку "Сделать фото" (Для выполнения работы потребуется веб-камера);
    \n8. Проверка:
    \nПровести визуальную оценку точности распознавания цифры нейронной сетью;
    \n9. Корректировка:
    \nС помощью инструментов, предоставленных программой, скорректировать изображение и распознать;
    \n10. Приложение Streamlit:
    \nОформление микросервиса Streamlit, выгрузка на сервер: проводится сотрудником лаборатории, используется студентами РАНХиГС
    \nС использованием библиотеки [streamlit](https://docs.streamlit.io/library/get-started)
    """)

with st.expander("Краткое описание искусственной нейронной сети и ее работы."):
            st.write('Искусственная нейронная сеть - это математическая модель настоящей нейронной сети,'  
                     'то есть мозга. На практике, это обучаемый под требуемую задачу инструмент.' 
                     'Искусственная нейронная сеть представляет собой набор матриц, с которыми работают по законам линейной алгебры. '
                     'Тем не менее, проще представить её как набор слоёв нейронов, связанных между собой '
                     'за счёт входных и выходных связей.')
            st.image('/app/laboratory4/pictures/fully_connected_NN.png', caption='Пример построения нейронной сети')
            st.write('Различают внешние слои - входной и выходной, и внутренние, находящиеся между ними. '
                     'У каждого отдельного нейрона, например, перцептрона, может быть несколько входных связей, у каждой из связей - свой множитель усиления' 
                     '(ослабления) влияния связи - весовой коэффициент, или вес. На выходе нейрона действует функция активации, за счёт нелинейностей ' 
                     'функций активации и подбора параметров-весов на входе нейрона, нейронная сеть и может обучаться.')
            st.image('/app/laboratory4/pictures/activation_functions.png',caption='Набор активационных функций')



st.markdown('''<h1 style='text-align: center; color: black;'
            >Задача лабораторной работы.</h1>''', 
            unsafe_allow_html=True)
st.write('  Возможность распознавать образы является одним из признаков интеллекта. Для компьютеров это уже не является сложной задачей.'
         'В данной работе Вам предстоит проверить насколько хорошо обучена нейронная сеть распознавать рукописные цифры. '
         'Это может пригодиться для создания программ прочтения и перевода в печатный текст рукописей или рецептов врача.')
    

st.write('Пункт 1.')
st.write('Возьмите любую из предложенных цифр.'
         'Эта цифра похожа на цифры обучающего набора, в чём можете убедиться, сравнив её '
         'с цифрами образцового набора на экране.')
st.image('/app/laboratory4/pictures/digits.png')

st.write('Пункт 2.'):
st.write('Поднесите цифру к видеокамере так, чтобы она занимала большую часть экрана видео в '
         'окошке, располагалась в центре и была хорошо освещена.')

st.write('Пункт 3.'):
st.write('Другой рукой возьмите мышь и щёлкните на кнопку под изображением, полученным с камеры, чтобы сделать снимок цифры.')

img_file_buffer = st.camera_input("Take a picture")
if img_file_buffer is not None:
            img = Image.open(img_file_buffer)
            img_array = np.array(img)
            img_height, img_width = img_array.shape[0], img_array.shape[1]
            img_center = int(img_width / 2)
            left_border = int(img_center - img_height / 2)
            right_border = int(img_center + img_height / 2)
            img_array1 = img_array[:, left_border:right_border, :]
            im = Image.fromarray(img_array1)
            im.save(file_path)
            
            
with st.expander('Пункт 5.'):
    st.write('Зарисуй полученное изображение чёрно-белой цифры из окошка в бланк отчёта. '
             'Необходимо на рисунке отобразить возникшие недочёты изображения цифры, например, пропуски. Чтобы'
             ' не зарисовывать всё чёрное пространство, рекомендуется изобразить ручкой цифру на белом фоне '
             'листа бланка отчёта.')

with st.expander('Пункт 6.'):
    st.write('Нажми на кнопку распознавания, запиши результат.')
    isbutton1 = st.button('Распознать')
    col3, col4 = st.columns(2)
    with col3:      
              st.write('Вот что увидела нейронная сеть.')
              if isbutton1:
                          image11 = Image.open(file_path)
                          st.image(file_path) 
                          img11 = image11.resize((28, 28), Image.LANCZOS)   
                          img11.save(file_path) 
                          
                          #img12 = img11.convert("L")
                          #imgData = np.asarray(img12)
                          #step_lobe = .4
                          #mid_img_color = np.sum(imgData) / imgData.size
                          #min_img_color = imgData.min()
                          
                          
                          #THRESHOLD_VALUE = int(mid_img_color - (mid_img_color - min_img_color) * step_lobe)
                          #thresholdedData = (imgData < THRESHOLD_VALUE) * 1.0
                          imgData1 = np.expand_dims(np.asarray(img11.convert("L")), axis=0)

                        
    with col4:
              st.write('Она распознала это как...')
              if isbutton1:
                          y_predict1 = model_2d.predict(imgData1) 
                          y_maxarg = np.argmax(y_predict1, axis=1)
                          st.subheader(int(y_maxarg))

with st.expander('Пункт 7.'):
    st.write('Включи коррекцию яркости. Посмотри, улучшило ли это изображение негатива цифры.'
             ' Зарисуй результат, как указано выше.')
    col5,col6 = st.columns(2)
    with col5:
              value_sli = st.slider('Коррекция яркости', 0.0, 100.0, 50.0)
    with col6:
              st.write('Яркость',value_sli)
              image111 = Image.open(file_path)
              enhancer = ImageEnhance.Brightness(image111)
            
              factor = 2*value_sli / 100 #фактор изменения
                
              im_output = enhancer.enhance(factor)
              im_output.save(file_path)
              #img111 = image111.resize((28, 28), Image.LANCZOS)     
              #img121 = img111.convert("L")
              #imgData = np.asarray(img121)
              #step_lobe = value_sli / 100
              #mid_img_color = np.sum(imgData) / imgData.size
              #min_img_color = imgData.min()
              #THRESHOLD_VALUE = (mid_img_color - (mid_img_color - min_img_color) * step_lobe)
              #thresholdedData = (imgData < THRESHOLD_VALUE) * 1.0
              #imgData1 = np.expand_dims(thresholdedData, axis=0)
              #st.write(imgData1.shape)
              #tgt1 = np.squeeze(imgData1)
              #st.write(tgt1.shape)
              #im111 = Image.fromarray(tgt1)
              #st.write(imgData1)
              #im111.save(file_path)
              st.image(file_path)
              #y_predict1 = model_2d.predict(imgData1) 
              #y_maxarg = np.argmax(y_predict1, axis=1)
              #st.subheader(int(y_maxarg))

with st.expander('Пункт 8.'):
    st.write('Нажми на кнопку распознавания, запиши результат.')
    isbutton2 = st.button('Распознать еще картнку')
    col7,col8 = st.columns(2)
    with col7:
             if isbutton2:
                   st.image(file_path)
    with col8:
             if isbutton2:
                   image112 = Image.open(file_path)
                   img111 = image112.resize((28, 28), Image.LANCZOS)  
                   img121 = img111.convert("L")
                   imgData = np.asarray(img121)
                   step_lobe = value_sli / 100
                   mid_img_color = np.sum(imgData) / imgData.size
                   min_img_color = imgData.min()
                   THRESHOLD_VALUE = (mid_img_color - (mid_img_color - min_img_color) * step_lobe)
                   thresholdedData = (imgData < THRESHOLD_VALUE) * 1.0
                   imgData1 = np.expand_dims(thresholdedData, axis=0)
                   y_predict1 = model_2d.predict(imgData1)
                   y_maxarg = np.argmax(y_predict1, axis=1)
                   st.subheader(int(y_maxarg))
                   
              

with st.expander('Пункт 9.'):
    st.write('Включи фильтр Гаусса, если такая кнопка есть, нажми на кнопку распознавания, запиши результат.')
    col9,col10 = st.columns(2)
    with col9:
            value_gaus = st.slider('Фильтр Гаусса', 0, 10, 0)
    with col10:
            st.write('Фильтр Гаусса',value_gaus)
            image222 = Image.open(file_path)
            im2 = image222.filter(ImageFilter.GaussianBlur(radius = value_gaus))
            im2.save(file_path)
            st.image(file_path)
with st.expander('Пункт 10'):
    st.write('Попробуем теперь еще раз распознать картинку.')
    isbutton3 = st.button('Распознать картнку еще раз')
    col11,col12 = st.columns(2)
    with col11:
            if isbutton3:
                   st.image(file_path)
    with col12:
            if isbutton3:
                   image333 = Image.open(file_path)
                   img333 = image333.resize((28, 28), Image.LANCZOS) 
                   img334 = img333.convert("L")
                   imgData4 = np.asarray(img334) 
                   step_lobe = value_sli / 100
                   mid_img_color = np.sum(imgData4) / imgData4.size
                   min_img_color = imgData4.min()
                   THRESHOLD_VALUE = (mid_img_color - (mid_img_color - min_img_color) * step_lobe)
                   thresholdedData = (imgData4 < THRESHOLD_VALUE) * 1.0
                   imgData5 = np.expand_dims(thresholdedData, axis=0)
                   y_predict2 = model_2d.predict(imgData5)
                   y_maxarg2 = np.argmax(y_predict2, axis=1)
                   st.subheader(int(y_maxarg2)) 
                    
with st.expander('Пункт 11.'):
    st.write('Сделай выводы, какие именно фильтры и как влияют на результат эксперимента')
with st.expander('Пункт 12.'):
    st.write('Посмотрим как "видит" картинку нейронная сеть')
    col13,col14 = st.columns(2)
    with col13:
            value_thres = st.slider('Порог отсечки', 0, 100, 50)
    with col14:
            st.write('Порог отсечки',value_thres)
            image444 = Image.open(file_path)
            #i1 = image444.resize((28, 28), Image.LANCZOS) 
            i2 = image444.convert("L")
            i3 = np.asarray(i2)
            step_lobe = value_thres / 100
            mid_img_color = np.sum(i3) / i3.size
            min_img_color = i3.min()
            THRESHOLD_VALUE = (mid_img_color - (mid_img_color - min_img_color) * step_lobe)
            thresholdedData = (i3 < THRESHOLD_VALUE) * 255.0
            imm1 = Image.fromarray(thresholdedData)
            imm1 = imm1.convert("L")
            imm1.save(file_path)
            st.write(imm1) 
            st.image(file_path)
            #i4 = np.expand_dims(thresholdedData, axis=0)
            #imm1 = Image.fromarray(i4) 
            #imm1.save(file_path)
            #st.image(file_path)
            
with st.expander('Пожелания и замечания'):                
    st.write('https://docs.google.com/spreadsheets/d/1TuGgZsT2cwAIlNr80LdVn4UFPHyEePEiBE-JG6IQUT0/edit?usp=sharing')
