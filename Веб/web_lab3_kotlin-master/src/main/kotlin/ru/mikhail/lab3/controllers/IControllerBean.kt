package ru.mikhail.lab3.controllers

import ru.mikhail.lab3.dto.ResponseData

interface IControllerBean {
    var x: Float
    var y: Float
    var r: Float
    fun completeRequest()
    fun getResultList(): List<ResponseData>
}
