import streamlit as st

#import cv2
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
from tensorflow.keras.models import load_model

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

txt = st.text_area("Искусственная нейронная сеть", """Искусственная нейронная сеть - это математическая модель настоящей нейронной сети, 
    то есть мозга. На практике, это обучаемый под требуемую задачу инструмент.
    Искусственная нейронная сеть представляет собой набор матриц, с которыми работают по законам линейной алгебры. 
    Тем не менее, проще представить её как набор слоёв нейронов, связанных между собой 
    засчёт входных и выходных связей. Различают внешние слои - входной и выходной, и внутренние, находящиеся между ними.
    У каждого отдельного нейрона, например, перцептрона, может быть несколько входных связей, у каждой из связей - свой множитель усиления 
    (ослабления) влияния связи - весовой коэффициент, или вес. На выходе нейрона действует функция активации, засчёт нелинейностей 
    функций активации и подбора параметров-весов на входе нейрона, нейронная сеть и может обучаться.
    """)

header_names = ["Перцептрон", "Функции активации", "Полносвязная нейронная сеть", "Градиентный поиск минимума",
                "Скорость обучения (момент)",
                "Датасет рукописные цифры", "Картинка с рукописной цифрой", "Картинка составлена из точек",
                "Картинка хранится как массив",
                "Наша модель нейронной сети", "График точности", "График функции потерь", "Матрица ошибок"
                ]
subheader_names = ["Схема перцептрона:", "Функции активации", "Полносвязная нейронная сеть",
                   "Градиентный поиск минимума", "Скорость обучения (момент)",
                   "Датасет рукописные цифры", "Картинка с рукописной цифрой", "Картинка составлена из точек",
                   "Картинка хранится как массив",
                   "Наша модель нейронной сети", "График точности", "График функции потерь", "Матрица ошибок"
                   ]
file_names = ["perceptron", "activation_functions", "fully_connected_NN", "gradient_decay", "gradient_momentum",
              "digits", "one_digit", "digit28x28", "data_to_line",
              "model_2d", "accuracy", "categorical_crossentropy", "confusion_matrix_2d_model"
              ]
text_headers = ["Информация о перцептроне", "Функции активации", "Полносвязная нейронная сеть",
                "Градиентный поиск минимума", "Скорость обучения (момент)",
                "Датасет рукописные цифры", "Картинка с рукописной цифрой", "Картинка составлена из точек",
                "Картинка хранится как массив",
                "Наша модель нейронной сети", "График точности", "График функции потерь", "Матрица ошибок"
                ]
texts = ["""
        Перцептрон - математический аналог нейрона. Из них состоит простейшая ИНС.
        На выходе перцептрона получаем математическую функцию нескольких подаваемых в него данных.
        И сам перцептрон по сути - математическая функция.
        Важные понятия: вес, смещение, функция активации.
        Вес - это число, регулирующее, насколько сильно воспринимается воздействие с данного входа
        Смещение - дополнительная регулировка. Как в уравнении прямой, она позволяет нашей функции смещаться,
        чтобы она могла проходить не только через ноль.
        Функция активации добавляет перцептрону нелинейность на выходе, то есть зависимость выходных данных от входных
        становится не лежащей на одной прямой.
        Это позволяет сетям на основе перцептронов обучаться и находить сложные закономерности.
        """,
        """Функция активации добавляет перцептрону нелинейность на выходе, то есть зависимость выходных данных от входных
        становится не лежащей на одной прямой.
        Это позволяет сетям на основе перцептронов обучаться и находить сложные закономерности.
        """,
        """Полносвязная нейронная сеть - нейронная сеть, составленная из слоёв таких перцептронов.
        Было доказано, что для нахождения закономерности любой сложности необходимо не менее двух слоёв перцептронов.""", 
        """Нейронная сеть обучается, подбирая веса таким образом, чтобы сделать минимальной разность между предсказанными 
        значениями и правильными ответами. Для этого сравнения используют функцию потерь. Различные типы функции потерь вы узнаете позже. 
        Градиентный поиск минимума позволяет найти наиболее оптимальное решение путём подбора таких параметров нейронной сети,
        чтобы функция потерь была минимальной. 
        При этом, сам градиент - это вектор, указывающий направление локального максимума, а мы должны написать минус перед ним,
        чтобы двигаться в противоположном направлении - к локальному, то есть местному, минимуму. Нейронная сеть - 
        функция многих параметров, от которых в итоге зависит и значение функции потерь, 
        и не всегда градиент находит наилучший, глобальный минимум функции. Чаще находит локальный.""", 
        """Скорость обучения (или момент) позволяет изменить длину "шага" для градиента. Это позволяет более точно
        подстроить веса, т.к. позволяет найти минимальную разницу между предсказанными и правильными значениями при обучении""",
        """Датасет рукописные цифры позволяет обучить нейронную сеть "узнавать" по картинке, какая цифра написана.
        Он состоит из картинок, на которых изображены цифры, написанные вручную, как если мы пишем ручкой на бумаге.
        Для нейронной сети это достаточно трудная задача. Вспомним, как тяжело читать текст, написанный почерком врача.""", 
        """Картинка с рукописной цифрой - здесь приводим пример изображения.""", 
        """На самом деле, картинка составлена из точек - пикселей. То есть она состоит из отдельных окрашенных фрагментов, 
        а не цельная. Это цифра для нас, но для компьютера это просто картинка.""", 
        """Картинка хранится как массив данных, то есть в упорядоченной последовательности цифр.""",
        """Наша модель нейронной сети - выводим, какие слои составляют нейронную сеть. Также, приводятся цифры, которые
        обозначают размеры матриц, которые используются в нейронной сети. Входной размер равен размеру картинки с цифрой, 
        Выходной - количеству классов, которые пытаемся распознать. На самом деле, вся нейронная сеть - математическая модель, 
        реализованная на операциях умножения и сложения матриц.""", 
        '''Точность (accuracy) - это метрика, равная отношению всех верных ответов к общему количеству ответов, умноженному на 100.
        Максимальное значение точности, когда все ответы правильные - 100 %, минимальное - 0 %. Это хорошая метрика, но она может 
        быть использована для оценки только если количество примеров каждого класса в выборке поровну или близко к одинаковому.
        График зависимости точности от количества эпох обучения позволяет увидеть и контролировать визуально обучение сети. 
        В идеале, точность на проверочной(валидационной) выборке 
        должна быть такая же, как на обучающей. Но нейронная сеть "ленится" узнавать закономерности и часто пытается просто заучить 
        обучающие данные, а не понять функции и законы, которым эти данные подчиняются. Это называется переобучением. На этом графике видно, 
        что точность на проверочной выборке данных меньше, чем на обучающей. Точность может даже падать при переобучении.''', 
        '''Функция потерь "категориальная перекрёстная энтропия" может быть рассчитана по приведённой ниже формуле. Чем она меньше, тем меньше 
        ошибается нейронная сеть. График функции потерь приведён на рисунке. Видно, что для проверочной выборки, график поднимается вправо, что 
        свидетельствует о переобучении.''', 
        """Матрица ошибок (confusion matrix) позволяет увидеть, на каких именно примерах данных (в данном случае, картинках каких именно цифр) 
        нейронная сеть или алгоритм более всего ошибается. Например, алгоритм может принять тройку за восьмёрку или девятку, особенно если
        они написаны неразборчиво или смещены. В ячейках записано количество ошибок данного типа. По диагонали матрицы находится число 
        верно распознанных цифр указанного на оси под квадратиком класса."""
        ]

file_path = '/app/laboratory3/'

for header_name, subheader_name, file_name, text_header, text in zip(header_names, subheader_names, file_names, text_headers, texts):
    # st.subheader(header_name)
    with st.expander(header_name):
        col11, col12 = st.columns(2)

        with col11:
            with st.container():
                st.subheader(subheader_name)

                #image = cv2.imread(file_path + file_name + '.png')
                #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = Image.open(file_path + file_name + '.png')
                st.image(image)

        with col12:
            with st.container():
                # st.subheader("Информация о перцептроне")

                txt = st.text_area(text_header, text, height=200)


def show_image(img):
    plt.imshow(Image.fromarray(img).convert('RGB')) #Отрисовка картинки .convert('RGB')
    plt.show()

model_2d = load_model('/app/laboratory3/mnist_2d.h5')    
file_path = '/app/laboratory3/your_file_image.png'
picture_all = '/app/laboratory3/realtrack1.jpg'

st.markdown('''<h1 style='text-align: center; color: black;'
            >Лабораторная работа "Распознавание рукописных цифр".</h1>''', 
            unsafe_allow_html=True)

with st.expander('Общая схема создания нейронной сети.'):
    st.image(picture_all)
    
with st.expander('Пункт 1.'):
    st.write('Разверни и прочитай все ячейки.')

with st.expander('Пункт 2.'):
    st.write('Возьми ту из предложенных цифр, которая подписана, что она хорошо распознаётся. '
             'Эта цифра похожа на цифры обучающего набора, в чём можешь убедиться, сравнив её '
             'с цифрами образцового набора на экране.')

col1, col2 = st.columns(2)
with col1:
            with st.expander('Пункт 3.'):
                        st.write('Поднеси цифру к видеокамере так, чтобы она занимала большую часть экрана видео в '
                                 'окошке, располагалась в центре и была хорошо освещена.')

            with st.expander('Пункт 4.'):
                        st.write('Другой рукой возьми мышь и щёлкни на кнопку, чтобы сделать снимок цифры.')

with col2: 
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
