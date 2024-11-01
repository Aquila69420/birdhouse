import React, { useState, useEffect } from 'react';
import { ethers } from 'ethers';
import { Box, Text, useColorModeValue } from "@chakra-ui/react";

const TokenBalanceCard = ({ walletId, tokenAddress }) => {
  const [balance, setBalance] = useState(null);
  const [error, setError] = useState(false);
  const bgColor = useColorModeValue("white", "gray.800");

  useEffect(() => {
    const fetchBalance = async () => {
      if (!window.ethereum) {
        console.error("Ethereum provider not found. Install MetaMask.");
        setError(true);
        return;
      }

      try {
        setError(false); // Reset error state before fetching

        // Use ethers BrowserProvider (v6) or Web3Provider (v5) depending on your version
        const provider = new ethers.BrowserProvider(window.ethereum);
        
        // ABI for balanceOf function in ERC-20 tokens
        const tokenAbi = ["function balanceOf(address owner) view returns (uint256)"];
        
        // Create a contract instance
        const tokenContract = new ethers.Contract(tokenAddress, tokenAbi, provider);
        
        // Fetch balance and format to 18 decimals
        const balanceBigNumber = await tokenContract.balanceOf(walletId);
        setBalance(ethers.formatUnits(balanceBigNumber, 18)); // Adjust 18 based on your token's decimals
      } catch (error) {
        console.error("Error fetching balance:", error);
        setError(true); // Set error state if fetching fails
      }
    };

    if (walletId && tokenAddress) {
      fetchBalance();
    }
  }, [walletId, tokenAddress]);

  return (
    <Box
      bg={bgColor}
      borderRadius="lg"
      boxShadow="md"
      p="6"
      display="flex"
      flexDirection="column"
      alignItems="center"
      justifyContent="center"
      textAlign="center"
    >
      <Text fontSize="lg" fontWeight="bold" color="gray.700" mb="2">
        Current Tokens
      </Text>
      <Text fontSize="2xl" fontWeight="bold" color="blue.500">
        {error ? "Invalid wallet" : balance ? `${balance} $FML` : "Loading..."}
      </Text>
    </Box>
  );
};

export default TokenBalanceCard;
