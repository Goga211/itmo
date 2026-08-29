package ru.iot.project

import org.springframework.context.annotation.Bean
import org.springframework.context.annotation.Configuration
import org.springframework.integration.channel.DirectChannel
import org.springframework.integration.core.MessageProducer
import org.springframework.integration.mqtt.core.DefaultMqttPahoClientFactory
import org.springframework.integration.mqtt.inbound.MqttPahoMessageDrivenChannelAdapter
import org.springframework.messaging.MessageChannel

@Configuration
class MqttInboundConfig(
    private val mqttClientFactory: DefaultMqttPahoClientFactory
) {

    @Bean
    fun mqttInputChannel(): MessageChannel =
        DirectChannel()

    @Bean
    fun inbound(): MessageProducer {
        val adapter = MqttPahoMessageDrivenChannelAdapter(
            "backend-client",
            mqttClientFactory,
            "sensor/#"   // 👈 ВАЖНО!
        )

        adapter.setOutputChannel(mqttInputChannel())
        return adapter
    }
}