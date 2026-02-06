package ru.mikhail.lab3.services

import jakarta.enterprise.context.ApplicationScoped
import jakarta.inject.Inject
import jakarta.inject.Named
import ru.mikhail.lab3.DotChecker.checkDot
import ru.mikhail.lab3.dto.RequestData
import ru.mikhail.lab3.dbobjects.Result
import ru.mikhail.lab3.repositories.ResultDAO
import java.sql.Timestamp
import java.time.LocalDateTime

@Named("resultServiceBean")
@ApplicationScoped
open class ResultService {

    companion object { private const val MIN_EXECUTION_TIME_NS = 1L }

    @Inject
    private lateinit var resultDAO: ResultDAO

    open fun findResult(id: Int): Result? =
        resultDAO.findById(id)

    open fun saveResult(result: Result) {
        resultDAO.save(result)
    }

    open fun updateResult(result: Result) {
        resultDAO.update(result)
    }

    open fun deleteResult(result: Result) {
        resultDAO.delete(result)
    }

    open fun findAllResults(): List<Result> =
        resultDAO.findAll()

    open fun completeRequest(requestData: RequestData) {
        val result = checkAndCalculatePoint(requestData)
        resultDAO.save(result)
    }

    open fun checkAndCalculatePoint(requestData: RequestData): Result {
        val start = System.nanoTime()
        val hit = checkDot(requestData.x, requestData.y, requestData.r)
        val dur = maxOf(System.nanoTime() - start, MIN_EXECUTION_TIME_NS)
        val now = Timestamp.valueOf(LocalDateTime.now())
        return Result(
            x = requestData.x,
            y = requestData.y,
            r = requestData.r,
            result = hit,
            executionTime = dur,
            nowTime = now
        )
    }
}
