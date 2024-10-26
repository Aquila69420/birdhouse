// ProfileInfoCard.js

import React from 'react';
import { Box, Avatar, Text, Heading, useColorModeValue } from "@chakra-ui/react";

const ProfileInfoCard = ({ name, walletId, avatarSrc }) => {
  const bgColor = useColorModeValue("white", "gray.800");

  return (
    <Box
      bg={bgColor}
      borderRadius="lg"
      boxShadow="md"
      p="6"
      textAlign="center"
    >
      <Avatar size="xl" src={avatarSrc} mb="4" />
      <Heading fontSize="lg" fontWeight="bold">
        {name}
      </Heading>
      <Text fontSize="sm" color="gray.500" mt="2">
        Wallet ID: {walletId}
      </Text>
    </Box>
  );
};

export default ProfileInfoCard;
