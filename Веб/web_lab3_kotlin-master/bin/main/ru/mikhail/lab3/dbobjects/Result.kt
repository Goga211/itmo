package ru.mikhail.lab3.dbobjects

import jakarta.persistence.*
import java.sql.Timestamp

@Entity
@Table(name = "result")
open class Result {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    open var id: Int? = null

    open var x: Float = 0f

    open var y: Float = 0f

    open var r: Float = 0f

    open var result: Boolean = false

    @Column(name = "execution_time")
    open var executionTime: Long = 0L

    @Column(name = "now_time")
    open var nowTime: Timestamp = Timestamp(System.currentTimeMillis())

    constructor()

    constructor(
        x: Float,
        y: Float,
        r: Float,
        result: Boolean,
        executionTime: Long,
        nowTime: Timestamp
    ) {
        this.x = x
        this.y = y
        this.r = r
        this.result = result
        this.executionTime = executionTime
        this.nowTime = nowTime
    }
}
