pipeline {
  agent {
    docker {
      image 'vivado-el7:3'
      args  '-v /data/Xilinx:/data/Xilinx'
    }
  }
  options {
    timeout(time: 6, unit: 'HOURS')
  }

  stages {
    stage('Keras to HLS') {
      steps {
        dir(path: 'test') {
          sh '''#!/bin/bash --login
              conda activate hls4ml-py38
              pip install tensorflow pyparsing
              pip install -U ../ --user
              ./convert-keras-models.sh -x -f keras-models.txt
              pip uninstall hls4ml -y'''
        }
      }
    }
    stage('C Simulation') {
      parallel {
        stage('2019.2') {
          when {
            allOf {
              environment name: 'USE_VIVADO_2019', value: '1';
              environment name: 'TEST_SIMULATION', value: '1'
            }
          }
          steps {
            dir(path: 'test') {
              sh '''#!/bin/bash
                 ./build-prj.sh -i /data/Xilinx -v 2019.2 -c -p 2'''
            }
          }
        }
        stage('2020.1') {
          when {
            allOf {
              environment name: 'USE_VIVADO_2020', value: '1';
              environment name: 'TEST_SIMULATION', value: '1'
            }
          }
          steps {
            dir(path: 'test') {
              sh '''#!/bin/bash
                 ./build-prj.sh -i /data/Xilinx -v 2020.1 -c -p 2'''
            }
          }
        }
      }
    }
    stage('C/RTL Synthesis') {
      parallel {
        stage('2019.2') {
          when {
            allOf {
              environment name: 'USE_VIVADO_2019', value: '1';
              environment name: 'TEST_SYNTHESIS', value: '1'
            }
          }
          steps {
            dir(path: 'test') {
              sh '''#!/bin/bash
                 ./build-prj.sh -i /data/Xilinx -v 2019.2 -s -r -p 2'''
            }
          }
        }
        stage('2020.1') {
          when {
            allOf {
              environment name: 'USE_VIVADO_2020', value: '1';
              environment name: 'TEST_SYNTHESIS', value: '1'
            }
          }
          steps {
            dir(path: 'test') {
              sh '''#!/bin/bash
                 ./build-prj.sh -i /data/Xilinx -v 2020.1 -s -r -p 2'''
            }
          }
        }
      }
    }
    stage('Report') {
      when {
        environment name: 'TEST_SYNTHESIS', value: '1'
      }
      steps {
        dir(path: 'test') {
          sh '''#!/bin/bash
             ./gather-reports.sh -b | tee report.rpt'''
        }
        archiveArtifacts artifacts: 'test/report.rpt', fingerprint: true
      }
    }
  }
  post {
    always {
      dir(path: 'test') {
          sh '''#!/bin/bash
             ./cleanup.sh'''
      }
    }
  }
}
