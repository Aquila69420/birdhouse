// PastTasks.js

import React from 'react';
import { Box, Text, Stack, CircularProgress, CircularProgressLabel, Button, useColorModeValue } from "@chakra-ui/react";

// Sample JSON data for past tasks
const pastTasksData = [
  {
    taskId: 21,
    title: "Image Classification",
    status: "Completed",
    stake: 10,
    availableToClaim: 5,
    dailyReward: 0.5,
    completionDate: "2024-10-20",
    progress: 100
  },
  {
    taskId: 22,
    title: "Sentiment Analysis",
    status: "Active",
    stake: 20,
    availableToClaim: 10,
    dailyReward: 1.2,
    completionDate: "2024-10-21",
    progress: 75
  },
  {
    taskId: 23,
    title: "Object Detection",
    status: "Completed",
    stake: 15,
    availableToClaim: 7.5,
    dailyReward: 0.8,
    completionDate: "2024-10-22",
    progress: 100
  }
];

const PastTasks = () => {
  // Set colors for light mode
  const bgColor = useColorModeValue("white", "gray.800");
  const textColor = useColorModeValue("gray.800", "white");
  const secondaryTextColor = useColorModeValue("gray.600", "gray.400");

  return (
    <Stack spacing="4">
      {pastTasksData.map(task => (
        <Box
          key={task.taskId}
          bg={bgColor}
          p="4"
          borderRadius="lg" // Ensures consistent corner radius with ComplexTable
          boxShadow="md"
          display="flex"
          flexDirection="column"
          alignItems="center"
          textAlign="center"
        >
          <CircularProgress
            value={task.progress}
            size="60px"
            color={task.status === "Completed" ? "green.400" : "blue.400"}
            mb="2"
          >
            <CircularProgressLabel fontSize="xs">
              {task.progress}%
            </CircularProgressLabel>
          </CircularProgress>

          <Text fontWeight="bold" fontSize="md" color={textColor}>
            {task.title}
          </Text>
          <Text fontSize="sm" color={secondaryTextColor}>
            Task ID: {task.taskId}
          </Text>
          <Text fontSize="sm" color={secondaryTextColor}>
            Status: {task.status}
          </Text>
          <Text fontSize="sm" color={secondaryTextColor}>
            Completion Date: {task.completionDate}
          </Text>
          <Text fontSize="sm" color={secondaryTextColor} mt="2">
            Your Stake: {task.stake} FML
          </Text>
          {task.dailyReward && (
            <Text fontSize="sm" color={secondaryTextColor}>
              Daily Reward: {task.dailyReward} FML
            </Text>
          )}
          <Text fontSize="sm" color={secondaryTextColor}>
            Available to Claim: {task.availableToClaim} FML
          </Text>

          <Button
            mt="4"
            colorScheme="blue"
            size="sm"
            width="full"
            isDisabled={task.status !== "Completed"}
          >
            Withdraw FML
          </Button>
        </Box>
      ))}
    </Stack>
  );
};

export default PastTasks;
