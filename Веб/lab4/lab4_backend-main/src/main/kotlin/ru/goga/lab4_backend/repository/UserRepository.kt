package ru.goga.lab4_backend.repository

import org.springframework.data.jpa.repository.JpaRepository
import org.springframework.stereotype.Repository
import ru.goga.lab4_backend.dbobjects.User

@Repository
interface UserRepository : JpaRepository<User, Long> {
    fun findUserByUsername(username: String?): User?
    fun existsByUsername(username: String): Boolean


}