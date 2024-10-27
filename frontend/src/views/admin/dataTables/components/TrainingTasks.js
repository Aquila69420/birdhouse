// TrainingTasks.js

import React from "react";
import { Box, Text } from "@chakra-ui/react";
import { columnsDataComplex } from "views/admin/dataTables/variables/columnsData";
import tableDataComplex from "views/admin/dataTables/variables/tableDataComplex.json";
import ComplexTable from "views/admin/dataTables/components/ComplexTable-Validate"; // Use the existing ComplexTable component

const TrainingTasks = ({ onTaskSelect }) => {
  return (
    <Box>
      <Text fontSize="lg" fontWeight="600" mb="4">
        Current Training Tasks
      </Text>
      <ComplexTable
        columnsData={columnsDataComplex}
        tableData={tableDataComplex}
        onTaskSelect={onTaskSelect} // Pass selected task to show additional actions
      />
    </Box>
  );
};

export default TrainingTasks;
