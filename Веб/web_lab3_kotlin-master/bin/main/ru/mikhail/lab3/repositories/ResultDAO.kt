package ru.mikhail.lab3.repositories

import jakarta.annotation.PostConstruct
import jakarta.annotation.PreDestroy
import jakarta.enterprise.context.ApplicationScoped
import jakarta.inject.Named
import jakarta.persistence.EntityManager
import jakarta.persistence.EntityManagerFactory
import jakarta.persistence.Persistence
import ru.mikhail.lab3.dbobjects.Result

@Named("resultDAOBean")
@ApplicationScoped
open class ResultDAO {

    private lateinit var emf: EntityManagerFactory

    @PostConstruct
    open fun init() {
        emf = Persistence.createEntityManagerFactory("lab3PU")
    }

    @PreDestroy
    open fun destroy() {
        if (::emf.isInitialized) {
            emf.close()
        }
    }

    private fun getEntityManager(): EntityManager {
        return emf.createEntityManager()
    }

    open fun findById(id: Int): Result? {
        val em = getEntityManager()
        return try {
            em.find(Result::class.java, id)
        } catch (e: Exception) {
            e.printStackTrace()
            null
        } finally {
            em.close()
        }
    }

    open fun findAll(): List<Result> {
        val em = getEntityManager()
        return try {
            em.createQuery("SELECT r FROM Result r ORDER BY r.nowTime DESC", Result::class.java)
                .resultList
        } catch (e: Exception) {
            e.printStackTrace()
            emptyList()
        } finally {
            em.close()
        }
    }

    open fun save(result: Result) {
        val em = getEntityManager()
        val tx = em.transaction
        try {
            tx.begin()
            em.persist(result)
            tx.commit()
        } catch (e: Exception) {
            if (tx.isActive) {
                tx.rollback()
            }
            e.printStackTrace()
            throw e
        } finally {
            em.close()
        }
    }

    open fun update(result: Result): Result {
        val em = getEntityManager()
        val tx = em.transaction
        return try {
            tx.begin()
            val merged = em.merge(result)
            tx.commit()
            merged
        } catch (e: Exception) {
            if (tx.isActive) {
                tx.rollback()
            }
            e.printStackTrace()
            throw e
        } finally {
            em.close()
        }
    }

    open fun delete(result: Result) {
        val em = getEntityManager()
        val tx = em.transaction
        try {
            tx.begin()
            val managed = if (em.contains(result)) result else em.merge(result)
            em.remove(managed)
            tx.commit()
        } catch (e: Exception) {
            if (tx.isActive) {
                tx.rollback()
            }
            e.printStackTrace()
            throw e
        } finally {
            em.close()
        }
    }
}
