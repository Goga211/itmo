package ru.mikhail.lab3.controllers

import jakarta.enterprise.context.SessionScoped
import jakarta.inject.Inject
import jakarta.inject.Named
import ru.mikhail.lab3.DataMapper
import ru.mikhail.lab3.dto.RequestData
import ru.mikhail.lab3.dto.ResponseData
import ru.mikhail.lab3.services.ResultService
import java.io.Serializable
import java.util.logging.Logger

@Named("controllerBean")
@SessionScoped
open class ControllerBean : IControllerBean, Serializable {

    private val logger: Logger = Logger.getLogger(ControllerBean::class.java.name)

    // Без Bean Validation — диапазоны контролируем на фронте
    override var x: Float = 0f      // [-2; 2] выбирается кнопками
    override var y: Float = 0f      // [-3; 3] валидируется через <f:validateDoubleRange>
    override var r: Float = 3.5f    // R от 2 до 5, стартовое значение по вкусу

    @Inject
    private lateinit var resultService: ResultService

    @Inject
    private lateinit var dataMapper: DataMapper

    @Inject
    private lateinit var resultsBean: ResultsBean

    override fun completeRequest() {
        logger.info("=== NEW POINT RECEIVED FROM FRONT ===")
        logger.info("X = $x, Y = $y, R = $r")

        resultService.completeRequest(RequestData(x, y, r))
        resultsBean.refresh()

        val last = resultsBean.getResultList().firstOrNull()
        if (last != null) {
            logger.info("Saved point -> hit=${last.result}, time=${last.executionTime} ns, now=${last.nowTime}")
        }

        val compact = resultsBean.getResultList().joinToString("; ") {
            "[x=${it.x}, y=${it.y}, r=${it.r}, result=${it.result}]"
        }
        logger.info("Current points in session: $compact")
        logger.info("====================================")
    }

    override fun getResultList(): List<ResponseData> {
        return resultsBean.getResultList()
    }
}
