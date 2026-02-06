import {createSlice} from "@reduxjs/toolkit";

export const usernameSlice = createSlice({
    name: 'usernameEditor',
    initialState:{
        username: ""
    },
    reducers:{
        editName: (state, action) =>{
            state.username = action.payload
        }
    }
})

export const {editName} = usernameSlice.actions

export default usernameSlice.reducer