package ru.mikhail.lab3

import jakarta.enterprise.context.ApplicationScoped
import jakarta.inject.Inject
import jakarta.inject.Named
import org.primefaces.event.SlideEndEvent
import java.io.Serializable
import java.util.logging.Logger
import ru.mikhail.lab3.controllers.ControllerBean

@Named("sliderListener")
@ApplicationScoped
open class SliderListener @Inject constructor(
    private val controllerBean: ControllerBean
) : Serializable {

    private val log: Logger = Logger.getLogger(SliderListener::class.java.name)

    open fun onSlideEnd(event: SlideEndEvent) {
        // PrimeFaces SlideEndEvent.value — это Number
        val value = event.value as Number
        val r = value.toFloat()

        log.info("Slider slideEnd: new R = $r")

        controllerBean.r = r
        // при желании можно дернуть что-то ещё у контроллера
        // controllerBean.onRadiusChange()
    }
}
