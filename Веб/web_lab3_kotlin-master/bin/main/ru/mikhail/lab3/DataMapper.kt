package ru.mikhail.lab3

import jakarta.enterprise.context.ApplicationScoped
import jakarta.inject.Named
import ru.mikhail.lab3.dbobjects.Result
import ru.mikhail.lab3.dto.ResponseData

@Named("dataMapper")
@ApplicationScoped
open class DataMapper {

    open fun getDataList(result: List<Result>): List<ResponseData> {
        return result.map { item ->
            ResponseData(item.x, item.y, item.r, item.result, item.executionTime, item.nowTime)
        }
    }
}