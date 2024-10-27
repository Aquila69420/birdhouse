// RunAutomation.jsx
import React from 'react';
import { Box, Text, Button, useToast } from "@chakra-ui/react";

export default function RunAutomation({ taskInfo }) {
  const toast = useToast();

  const handleRunScript = () => {
    toast({ title: "Running automation script...", status: "info", duration: 2000 });
    // Run automation script using taskInfo
  };

  return (
    <Box>
      <Text fontSize="xl" fontWeight="bold">Step 4: Run Automation Script</Text>
      <Button onClick={handleRunScript} colorScheme="green" mt={4}>Run Script</Button>
    </Box>
  );
}
