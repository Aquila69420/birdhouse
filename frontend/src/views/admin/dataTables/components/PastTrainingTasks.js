// PastTrainingTasks.js

import React from "react";
import { Box, SimpleGrid, VStack, Text, Button, Divider } from "@chakra-ui/react";

const pastTasksData = [
  { taskId: 1, status: "Completed", tokensStaked: 100, reward: 50 },
  { taskId: 2, status: "Completed", tokensStaked: 150, reward: 75 },
  // Add more data as needed
];

const PastTrainingTasks = () => {
  return (
    <Box p={4} bg="white" borderRadius="lg" boxShadow="md">
      <Text fontSize="lg" fontWeight="bold" mb="4">
        My Past Training Tasks
      </Text>
      <SimpleGrid columns={{ base: 1, md: 2 }} spacing={4}>
        {pastTasksData.map((task) => (
          <Box
            key={task.taskId}
            p={4}
            bg="gray.50"
            border="1px solid"
            borderColor="gray.200"
            borderRadius="lg"
            boxShadow="sm"
          >
            <VStack align="start" spacing={2}>
              <Text fontWeight="bold">Task {task.taskId}</Text>
              <Divider />
              <Text>Status: {task.status}</Text>
              <Text>Tokens Staked: {task.tokensStaked} FML</Text>
              <Text>Reward Earned: {task.reward} FML</Text>
              <Button size="sm" colorScheme="blue" variant="outline">
                View Details
              </Button>
            </VStack>
          </Box>
        ))}
      </SimpleGrid>
    </Box>
  );
};

export default PastTrainingTasks;
