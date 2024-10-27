// EnvironmentSetup.jsx
import React from 'react';
import { Box, Text, Button, VStack, useToast } from "@chakra-ui/react";

export default function EnvironmentSetup() {
  const toast = useToast();

  const handleCheckInstallation = (platform) => {
    if (platform === 'Windows') {
      toast({ title: "Checking WSL installation...", status: "info", duration: 2000 });
      // Run WSL installation check command here
    } else {
      toast({ title: "Checking Anaconda installation...", status: "info", duration: 2000 });
      // Run Anaconda installation check command here
    }
  };

  return (
    <Box>
      <Text fontSize="xl" fontWeight="bold">Step 1: Set Up Your Environment</Text>
      <VStack spacing={4} mt={4}>
        <Button onClick={() => handleCheckInstallation("Windows")}>Check WSL Installation (Windows)</Button>
        <Button onClick={() => handleCheckInstallation("Mac/Linux")}>Check Anaconda Installation (Mac/Linux)</Button>
      </VStack>
    </Box>
  );
}
