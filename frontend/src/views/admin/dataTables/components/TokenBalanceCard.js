// TokenBalanceCard.js

import React from 'react';
import { Box, Text, useColorModeValue } from "@chakra-ui/react";

const TokenBalanceCard = ({ tokens }) => {
  const bgColor = useColorModeValue("white", "gray.800");

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
        {tokens} $FML
      </Text>
    </Box>
  );
};

export default TokenBalanceCard;
