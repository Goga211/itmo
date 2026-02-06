package ru.goga.lab4_backend


import org.mapstruct.Mapper
import ru.goga.lab4_backend.dbobjects.Dot
import ru.goga.lab4_backend.data.responses.GetDotsResponse

@Mapper(componentModel = "spring")
interface DotMapper {

    fun toDto(dot: Dot): GetDotsResponse

    fun toEntity(getDotsResponse: GetDotsResponse): Dot
}