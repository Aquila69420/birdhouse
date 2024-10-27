import { createSlice, createAsyncThunk } from '@reduxjs/toolkit';

// Async thunk to fetch the FML token balance
export const fetchFmlTokens = createAsyncThunk('person/fetchFmlTokens', async (address) => {
  // Simulating an API call to fetch token balance
  const response = await new Promise((resolve) => {
    setTimeout(() => {
      const tokenBalance = 1000;  // Let's assume we get a balance of 1000 tokens
      resolve({ balance: tokenBalance });
    }, 1000);
  });
  return response.balance;
});

const personSlice = createSlice({
  name: 'person',
  initialState: {
    name: '',
    fmlTokens: 0,
    address: '',
    isLoggedIn: false
  },
  reducers: {
    setName: (state, action) => {
      state.name = action.payload;
    },

    setAddress: (state, action) => {
      state.address = action.payload;
    },
    setIsLoggedIn: (state, action) => {
        state.isLoggedIn = action.payload
    }
  },
  extraReducers: (builder) => {
    builder
      .addCase(fetchFmlTokens.pending, (state) => {
        state.status = 'loading';
      })
      .addCase(fetchFmlTokens.fulfilled, (state, action) => {
        state.status = 'succeeded';
        state.fmlTokens = action.payload;
      })
      .addCase(fetchFmlTokens.rejected, (state, action) => {
        state.status = 'failed';
        state.error = action.error.message;
      });
  },
});

// Export actions for use in components
export const { setName, setAddress, setIsLoggedIn } = personSlice.actions;

// Export the reducer to be used in the store
export default personSlice.reducer;
