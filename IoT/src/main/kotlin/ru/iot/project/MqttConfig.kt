package ru.iot.project

import org.eclipse.paho.client.mqttv3.MqttConnectOptions
import org.springframework.beans.factory.annotation.Value
import org.springframework.context.annotation.Bean
import org.springframework.context.annotation.Configuration
import org.springframework.integration.mqtt.core.DefaultMqttPahoClientFactory

@Configuration
class MqttConfig {

    @Value("\${mqtt.broker-url}")
    private lateinit var brokerUrl: String

    @Bean
    fun mqttClientFactory(): DefaultMqttPahoClientFactory {
        val factory = DefaultMqttPahoClientFactory()
        val options = MqttConnectOptions()
        options.serverURIs = arrayOf(brokerUrl)
        factory.connectionOptions = options
        return factory
    }
}