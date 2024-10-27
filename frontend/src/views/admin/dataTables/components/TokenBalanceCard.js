// TokenBalanceCard.js
import React, { useState, useEffect } from 'react';
import { ethers } from 'ethers';
import { Box, Text, useColorModeValue } from "@chakra-ui/react";

const TokenBalanceCard = ({ walletId, tokenAddress }) => {
  const [balance, setBalance] = useState(null);
  const bgColor = useColorModeValue("white", "gray.800");

  useEffect(() => {
    const fetchBalance = async () => {
      try {
        // Connect to Ethereum provider (e.g., MetaMask)
        const provider = new ethers.BrowserProvider(window.ethereum);
        
        // ABI for balanceOf function in ERC-20 tokens
        const tokenAbi = ["function balanceOf(address owner) view returns (uint256)"];
        
        // Create a contract instance
        const tokenContract = new ethers.Contract(tokenAddress, tokenAbi, provider);
        
        // Fetch balance and format to 18 decimals
        const balance = await tokenContract.balanceOf(walletId);
        setBalance(ethers.formatUnits(balance, 18)); // Adjust 18 based on your token's decimals
      } catch (error) {
        console.error("Error fetching balance:", error);
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
        {balance ? `${balance} $FML` : "Loading..."}
      </Text>
    </Box>
  );
};

export default TokenBalanceCard;
