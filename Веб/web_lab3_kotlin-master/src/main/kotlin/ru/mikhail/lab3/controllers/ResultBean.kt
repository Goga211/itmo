package ru.mikhail.lab3.controllers

import jakarta.annotation.PostConstruct
import jakarta.faces.view.ViewScoped
import jakarta.inject.Inject
import jakarta.inject.Named
import ru.mikhail.lab3.DataMapper
import ru.mikhail.lab3.dto.ResponseData
import ru.mikhail.lab3.services.ResultService
import java.io.Serializable

@Named("resultsBean")
@ViewScoped
open class ResultsBean : Serializable {

    @Inject
    private lateinit var resultService: ResultService

    @Inject
    private lateinit var dataMapper: DataMapper

    private lateinit var list: List<ResponseData>

    @PostConstruct
    open fun init() {
        refresh()
    }

    open fun refresh() {
        list = dataMapper.getDataList(resultService.findAllResults())
    }

    open fun getResultList(): List<ResponseData> = list
}
