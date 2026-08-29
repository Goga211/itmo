package ru.iot.project

import com.fasterxml.jackson.databind.ObjectMapper
import com.fasterxml.jackson.module.kotlin.registerKotlinModule
import org.springframework.integration.annotation.MessageEndpoint
import org.springframework.integration.annotation.ServiceActivator
import org.springframework.messaging.Message
import org.springframework.stereotype.Service

@MessageEndpoint
class MqttListener {

    private val objectMapper = ObjectMapper().registerKotlinModule()

    @ServiceActivator(inputChannel = "mqttInputChannel")
    fun handle(message: Message<String>) {
        val topic = message.headers["mqtt_receivedTopic"] as? String
        val payload = message.payload

        when (topic) {
            "sensor/max30102/bpm" -> {
                val bpm = objectMapper.readValue(payload, BpmPayload::class.java)
                println("BPM: window=${bpm.bpmWindow}, buffer=${bpm.bpmBuffer}, ir=${bpm.ir}, samples=${bpm.samples}")
            }
            "sensor/max30102/state" -> {
                println("State: $payload")
            }
            "sensor/max30102/raw" -> {
                println("Raw IR: $payload")
            }
            else -> println("Unknown topic=$topic payload=$payload")
        }
    }
}