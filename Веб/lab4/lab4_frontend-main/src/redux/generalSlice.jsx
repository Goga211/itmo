import {createSlice} from "@reduxjs/toolkit";

export const generalSlice = createSlice({
    name: 'tableEditor',
    initialState: {
        table: [],
        refreshTable: false,
        radius: 3,
        loading: false,
        scaleCounter: 0,
    },
    reducers: {
        editTable: (state, action) => {
            state.table = action.payload

        },
        triggerRefresh: (state) => {
            state.refreshTable = !state.refreshTable
        },
        updateGraph(state, action) {
            state.radius = action.payload
        },
        incrementScaleCounter: (state) => {
            state.scaleCounter += 1
        },
        resetScaleCounter: (state) => {
            state.scaleCounter = 0
        },
        setLoading: (state, action) => {
            state.loading = action.payload
        }
    }

})

export const {
    editTable,
    triggerRefresh,
    updateGraph,
    incrementScaleCounter,
    resetScaleCounter,
    setLoading
} = generalSlice.actions

export default generalSlice.reducer