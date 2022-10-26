**Данный репозиторий содержит материалы для решения учебной задачи**


**Задача:**


Необходимо разработать модель, которая будет "сравнивать" названия компаний среди двух столбцов таблицы, и выдавать являются ли они "дублями". Так же на базе лучшей модели реализовать "поисковый движок" по данному датасету.  Тестовых данных не будет, метрику качества для оценки модели предоставят позже.


**Решение:**
1. В папке `experiments` находится ноутбук(*прилепить ссылку*) с различными моделями и методами для формирования векторного представления текстовых данных. Для выбора модели, которая будет использоваться далее, было принято решение оченивать их по метрике ROC-AUC. Она отлично подходит под задачу классификации. Результаты можно видеть ниже в таблице.

<p align="center"><img src="./saves/models.png"\></p>

В качестве основого метода выбрали TF-IDF + LogReg, так как он показал лучшие значения метрики и был одним из самых быстрых по скорости обработки.

2. *Написать про нашу реализацию пайплайна*
3. *Описать как пользоваться нашим скриптом*
