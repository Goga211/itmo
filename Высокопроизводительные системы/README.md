# Высокопроизводительные системы

Групповой проект «Каршеринг» на JVM-стеке, который поэтапно развивается от монолита до микросервисов.
Требования курса: https://github.com/Discipliny/highload_systems

Каждая лабораторная лежит в своём репозитории и подключена сюда сабмодулем.

| Лаба | Репозиторий | Содержание |
|---|---|---|
| ЛР1 | [highload-lab1](https://github.com/Goga211/highload-lab1) | монолит на Kotlin и Spring Boot, PostgreSQL, Liquibase, Testcontainers |
| ЛР2 | [highload-lab2](https://github.com/Goga211/highload-lab2) | микросервисы: Eureka, Config Server, Gateway, Feign, Circuit Breaker, R2DBC |
| ЛР3 | [highload-lab3](https://github.com/Goga211/highload-lab3) | авторизация: Spring Security, JWT, ролевая модель |
| ЛР4 | [highload-lab4](https://github.com/Goga211/highload-lab4) | Kafka, уведомления, файловый сервис, Clean Architecture |

Клонировать вместе с лабами:

```bash
git clone --recurse-submodules https://github.com/Goga211/itmo.git
```

Подтянуть свежие версии лаб:

```bash
git submodule update --remote
```
