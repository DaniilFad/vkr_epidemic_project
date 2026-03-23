# VKR Epidemic Project

Гибридная система краткосрочного прогнозирования заболеваемости на уровне города на основе интеграции эпидемиологических признаков, цифровых следов, погодных и календарных факторов с интерпретацией через Explainable AI.

## О проекте

Этот репозиторий содержит код и артефакты магистерской ВКР, посвящённой разработке гибридной модели прогнозирования распространения инфекционных заболеваний.  
Прикладной кейс в текущей реализации — COVID-19, Москва, недельные данные.

Основная идея проекта — объединить в одной исследовательской системе:

- официальную эпидемиологическую статистику;
- цифровые следы (Yandex Wordstat);
- погодные данные;
- календарные признаки;
- автоматический подбор лагов;
- подавление медиашума;
- честкий walk-forward backtesting;
- интерпретацию прогноза через XAI.

## Цель

Разработать и экспериментально проверить систему краткосрочного прогнозирования заболеваемости и раннего предупреждения эпидемических вспышек на уровне города.

## Исследовательская гипотеза

Использование гибридной системы краткосрочного прогнозирования, интегрирующей эпидемиологические признаки, цифровые следы и Explainable AI, на горизонте 7 дней:

- улучшает качество прогноза не менее чем на 15% по MAPE;
- снижает RMSE не менее чем на 20% по сравнению с базовыми моделями прогноза;
- позволяет выявлять интерпретируемые факторы, влияющие на прогноз заболеваемости.

## Данные

В проекте используются следующие источники данных:

- официальная статистика COVID-19 по Москве;
- поисковые запросы Yandex Wordstat;
- погодные данные;
- календарные и праздничные признаки.

Структура сырых данных:

- data/raw/covid/
- data/raw/wordstat/
- data/raw/weather/
- data/raw/calendar/

Промежуточные и итоговые датасеты:

- data/interim/base_weekly_dataset.csv
- data/processed/modeling_dataset.csv
- data/processed/modeling_dataset_advanced.csv

## Архитектура пайплайна

Проект устроен как пошаговый исследовательский pipeline.

### Базовые этапы

1. Сбор и подготовка данных.
2. Агрегация к единой недельной шкале.
3. Построение базового датасета.
4. Создание лаговых, сезонных, погодных и календарных признаков.
5. Обучение baseline- и hybrid-моделей.
6. Robust-сравнение моделей на walk-forward backtesting.
7. Формирование финального прогноза и alerts.
8. Визуализация, XAI и интерпретация.

### Продвинутые этапы

9. Интуитивная визуализация single-origin прогноза.
10. Построение advanced-датасета:
   - горизонт-специфические лаги;
   - признаки роста Wordstat;
   - альтернативные target-представления (ratio, log-delta).
11. Сравнение advanced-моделей.
12. Визуальное сравнение naive / hybrid / advanced на большом окне backtest.
13. Формирование итоговой mixed-system.
14. Построение mixed XAI для финальных выбранных моделей.

## Структура проекта

```text
vkr_epidemic_project/
├── configs/
│   └── project_config.yaml
├── data/
│   ├── raw/
│   │   ├── calendar/
│   │   ├── covid/
│   │   ├── weather/
│   │   └── wordstat/
│   ├── interim/
│   │   └── base_weekly_dataset.csv
│   └── processed/
│       ├── modeling_dataset.csv
│       └── modeling_dataset_advanced.csv
├── reports/
│   ├── figures/
│   ├── predictions/
│   └── tables/
├── src/
│   ├── init.py
│   ├── aligned_plots.py
│   ├── build_dataset.py
│   ├── data_io.py
│   ├── features.py
│   ├── final_forecast.py
│   ├── intuitive_forecast_plot.py
│   ├── model_comparison_plots.py
│   ├── models.py
│   ├── models_advanced.py
│   └── xai_plots.py
├── aggregate_weather_rp5.py
├── check_step1.py
├── generate_calendar_features.py
├── make_templates.py
├── parse_moscow_covid_html.py
├── run_step2.py
├── run_step3.py
├── run_step4.py
├── run_step5.py
├── run_step6.py
├── run_step7.py
├── run_step8.py
├── run_step8_aligned.py
├── run_step9_single_origin_demo.py
├── run_step10_build_advanced_dataset.py
├── run_step11_compare_advanced.py
├── run_step12_compare_models_visual.py
├── run_step13_final_mixed_system.py
├── run_step14_mixed_xai.py
└── requirements.txt