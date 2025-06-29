'use client'

import { useState, useEffect, useRef } from 'react'
import Link from 'next/link'
import Image from 'next/image'
import { useDropzone } from 'react-dropzone'
import { motion, AnimatePresence } from 'framer-motion'
import { useTasks, useNotifications } from './providers'
import { 
  ArrowRight, 
  Upload, 
  LineChart, 
  Video, 
  Image as ImageIcon, 
  BarChart3, 
  Layers, 
  Zap,
  Tag,
  ThumbsUp,
  ThumbsDown,
  MessageSquare,
  Play,
  Pause,
  ChevronLeft,
  ChevronRight,
  ZoomIn,
  ZoomOut,
  Download,
  Share2
} from 'lucide-react'
import { formatDistanceToNow } from 'date-fns'

// Types
interface UploadState {
  isUploading: boolean
  progress: number
  file: File | null
}

interface PoseVisualizationProps {
  taskId: string
  frameIndex: number
  totalFrames: number
}

interface UserFeedback {
  taskId: string
  frameIndex: number
  rating: 'positive' | 'negative' | null
  tags: string[]
  comment: string
}

export default function HomePage() {
  const { tasks, createTask, isLoading } = useTasks()
  const { addNotification } = useNotifications()
  const [uploadState, setUploadState] = useState<UploadState>({
    isUploading: false,
    progress: 0,
    file: null
  })
  const [stats, setStats] = useState({
    totalTasks: 0,
    successfulTasks: 0,
    processingTime: 0,
    detectedPoses: 0
  })
  const [selectedTask, setSelectedTask] = useState<string | null>(null)
  const [selectedFrame, setSelectedFrame] = useState<number>(0)
  const [isPlaying, setIsPlaying] = useState<boolean>(false)
  const [zoomLevel, setZoomLevel] = useState<number>(1)
  const [userFeedback, setUserFeedback] = useState<UserFeedback[]>([])
  const [availableTags, setAvailableTags] = useState<string[]>([
    'Good Detection', 'Missing Joints', 'Wrong Pose', 'Occlusion', 'Multiple People', 'Lighting Issues'
  ])
  const [currentComment, setCurrentComment] = useState<string>('')
  const [selectedTags, setSelectedTags] = useState<string[]>([])
  
  // Animation player reference
  const animationRef = useRef<number | null>(null)

  // Update stats based on tasks
  useEffect(() => {
    if (tasks.length > 0) {
      const successful = tasks.filter(task => task.status === 'SUCCESS').length
      const avgTime = tasks
        .filter(task => task.status === 'SUCCESS' && task.started_at && task.finished_at)
        .map(task => {
          const start = new Date(task.started_at as string).getTime()
          const end = new Date(task.finished_at as string).getTime()
          return (end - start) / 1000 // seconds
        })
        .reduce((sum, time) => sum + time, 0) / (successful || 1)
      
      // This is a placeholder - in a real app, we'd get this from the task data
      const detectedPoses = successful * 15 // Assuming average of 15 poses per task
      
      setStats({
        totalTasks: tasks.length,
        successfulTasks: successful,
        processingTime: avgTime,
        detectedPoses
      })
    }
  }, [tasks])

  // Handle animation playback
  useEffect(() => {
    if (isPlaying && selectedTask) {
      // Get the task to determine total frames
      const task = tasks.find(t => t.id === selectedTask)
      if (!task) return
      
      // Assuming we have 30 frames (this would come from the task data in a real app)
      const totalFrames = 30
      
      // Animation loop
      const animate = () => {
        setSelectedFrame(prev => {
          const next = (prev + 1) % totalFrames
          return next
        })
        animationRef.current = requestAnimationFrame(animate)
      }
      
      animationRef.current = requestAnimationFrame(animate)
      
      // Cleanup
      return () => {
        if (animationRef.current) {
          cancelAnimationFrame(animationRef.current)
        }
      }
    }
  }, [isPlaying, selectedTask, tasks])

  // Dropzone configuration
  const { getRootProps, getInputProps, isDragActive, acceptedFiles } = useDropzone({
    accept: {
      'video/*': ['.mp4', '.avi', '.mov', '.mkv', '.webm'],
      'image/*': ['.jpg', '.jpeg', '.png', '.bmp']
    },
    maxFiles: 1,
    onDrop: (acceptedFiles) => {
      if (acceptedFiles.length > 0) {
        handleFileUpload(acceptedFiles[0])
      }
    }
  })

  // Handle file upload
  const handleFileUpload = async (file: File) => {
    setUploadState({
      isUploading: true,
      progress: 0,
      file
    })

    // Simulate progress updates
    const interval = setInterval(() => {
      setUploadState(prev => ({
        ...prev,
        progress: Math.min(prev.progress + 10, 90)
      }))
    }, 300)

    try {
      // Create task with file
      await createTask(file)
      
      // Complete progress
      clearInterval(interval)
      setUploadState(prev => ({
        ...prev,
        progress: 100
      }))
      
      // Reset after a delay
      setTimeout(() => {
        setUploadState({
          isUploading: false,
          progress: 0,
          file: null
        })
      }, 1000)
      
    } catch (error) {
      clearInterval(interval)
      setUploadState({
        isUploading: false,
        progress: 0,
        file: null
      })
      
      addNotification({
        type: 'error',
        title: 'Upload Failed',
        message: 'There was an error uploading your file. Please try again.'
      })
    }
  }

  // Handle user feedback submission
  const handleFeedbackSubmit = () => {
    if (!selectedTask || selectedTags.length === 0) return
    
    const newFeedback: UserFeedback = {
      taskId: selectedTask,
      frameIndex: selectedFrame,
      rating: null, // Will be set by thumbs up/down
      tags: selectedTags,
      comment: currentComment
    }
    
    setUserFeedback(prev => [...prev, newFeedback])
    setSelectedTags([])
    setCurrentComment('')
    
    addNotification({
      type: 'success',
      title: 'Feedback Submitted',
      message: 'Thank you for your feedback!'
    })
  }

  // Handle tag selection
  const toggleTag = (tag: string) => {
    if (selectedTags.includes(tag)) {
      setSelectedTags(selectedTags.filter(t => t !== tag))
    } else {
      setSelectedTags([...selectedTags, tag])
    }
  }

  // Features data
  const features = [
    {
      title: 'Multi-Person Detection',
      description: 'Accurately detect and track multiple people in complex scenes',
      icon: <Layers className="h-6 w-6 text-blue-500" />
    },
    {
      title: 'Real-Time Processing',
      description: 'Fast and efficient pose detection for videos and image sequences',
      icon: <Zap className="h-6 w-6 text-yellow-500" />
    },
    {
      title: 'Comprehensive Output',
      description: 'Get detailed JSON data, raw frames, and visualized overlays',
      icon: <BarChart3 className="h-6 w-6 text-green-500" />
    },
    {
      title: 'Advanced Visualization',
      description: 'Interactive tools to explore and analyze detected poses',
      icon: <LineChart className="h-6 w-6 text-purple-500" />
    }
  ]

  // Quick start steps
  const quickStartSteps = [
    {
      number: 1,
      title: 'Upload Media',
      description: 'Drag and drop your video or images into the upload area'
    },
    {
      number: 2,
      title: 'Configure Options',
      description: 'Customize detection parameters or use the default settings'
    },
    {
      number: 3,
      title: 'Process and Analyze',
      description: 'Let AlphaDetect process your media and view the results'
    }
  ]

  // Pose Visualization Component
  const PoseVisualization = ({ taskId, frameIndex, totalFrames = 30 }: PoseVisualizationProps) => {
    // In a real app, we would fetch the actual frame and pose data
    // For this demo, we'll use a placeholder image and simulate pose joints
    
    // Generate simulated pose data
    const generatePoseData = () => {
      // Simulated keypoints for a person
      const centerX = 200 + Math.sin(frameIndex * 0.1) * 20
      const centerY = 200 + Math.cos(frameIndex * 0.1) * 10
      
      // Basic human pose keypoints (simplified)
      return [
        { x: centerX, y: centerY - 90, confidence: 0.9 }, // Head
        { x: centerX, y: centerY - 60, confidence: 0.95 }, // Neck
        { x: centerX, y: centerY, confidence: 0.98 }, // Torso
        { x: centerX - 30, y: centerY - 40, confidence: 0.9 }, // Left shoulder
        { x: centerX + 30, y: centerY - 40, confidence: 0.9 }, // Right shoulder
        { x: centerX - 60, y: centerY - 20, confidence: 0.85 }, // Left elbow
        { x: centerX + 60, y: centerY - 20, confidence: 0.85 }, // Right elbow
        { x: centerX - 80, y: centerY, confidence: 0.8 }, // Left wrist
        { x: centerX + 80, y: centerY, confidence: 0.8 }, // Right wrist
        { x: centerX - 20, y: centerY + 50, confidence: 0.9 }, // Left hip
        { x: centerX + 20, y: centerY + 50, confidence: 0.9 }, // Right hip
        { x: centerX - 30, y: centerY + 120, confidence: 0.85 }, // Left knee
        { x: centerX + 30, y: centerY + 120, confidence: 0.85 }, // Right knee
        { x: centerX - 35, y: centerY + 190, confidence: 0.8 }, // Left ankle
        { x: centerX + 35, y: centerY + 190, confidence: 0.8 }, // Right ankle
      ]
    }
    
    // Connections between keypoints to draw skeleton
    const skeletonConnections = [
      [0, 1], // Head to neck
      [1, 2], // Neck to torso
      [1, 3], // Neck to left shoulder
      [1, 4], // Neck to right shoulder
      [3, 5], // Left shoulder to left elbow
      [4, 6], // Right shoulder to right elbow
      [5, 7], // Left elbow to left wrist
      [6, 8], // Right elbow to right wrist
      [2, 9], // Torso to left hip
      [2, 10], // Torso to right hip
      [9, 11], // Left hip to left knee
      [10, 12], // Right hip to right knee
      [11, 13], // Left knee to left ankle
      [12, 14], // Right knee to right ankle
    ]
    
    const poseData = generatePoseData()
    
    return (
      <div className="relative w-full h-[400px] bg-gray-100 dark:bg-gray-700 rounded-lg overflow-hidden">
        {/* Simulated frame image */}
        <div className="absolute inset-0 flex items-center justify-center">
          <div className="w-full h-full bg-gradient-to-br from-gray-200 to-gray-300 dark:from-gray-600 dark:to-gray-800">
            {/* Simulated person silhouette */}
            <div className="absolute inset-0 flex items-center justify-center">
              <div 
                className="relative"
                style={{ transform: `scale(${zoomLevel})` }}
              >
                {/* Draw skeleton connections */}
                <svg width="400" height="400" viewBox="0 0 400 400" className="absolute top-0 left-0">
                  {skeletonConnections.map((connection, i) => {
                    const [from, to] = connection
                    return (
                      <line
                        key={i}
                        x1={poseData[from].x}
                        y1={poseData[from].y}
                        x2={poseData[to].x}
                        y2={poseData[to].y}
                        stroke="#3b82f6"
                        strokeWidth="3"
                        strokeLinecap="round"
                      />
                    )
                  })}
                  
                  {/* Draw keypoints */}
                  {poseData.map((point, i) => (
                    <circle
                      key={i}
                      cx={point.x}
                      cy={point.y}
                      r={i === 0 ? 8 : 6}
                      fill={point.confidence > 0.9 ? "#22c55e" : point.confidence > 0.7 ? "#eab308" : "#ef4444"}
                      stroke="#ffffff"
                      strokeWidth="2"
                    />
                  ))}
                </svg>
                
                {/* Frame counter */}
                <div className="absolute top-4 right-4 bg-black/50 text-white px-2 py-1 rounded text-sm">
                  Frame: {frameIndex + 1} / {totalFrames}
                </div>
              </div>
            </div>
          </div>
        </div>
        
        {/* Controls overlay */}
        <div className="absolute bottom-0 left-0 right-0 bg-black/50 p-3 flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <button 
              onClick={() => setIsPlaying(!isPlaying)}
              className="p-2 rounded-full bg-white/20 hover:bg-white/30 text-white transition-colors"
            >
              {isPlaying ? <Pause size={16} /> : <Play size={16} />}
            </button>
            
            <button 
              onClick={() => setSelectedFrame(prev => Math.max(0, prev - 1))}
              className="p-2 rounded-full bg-white/20 hover:bg-white/30 text-white transition-colors"
              disabled={selectedFrame === 0}
            >
              <ChevronLeft size={16} />
            </button>
            
            <button 
              onClick={() => setSelectedFrame(prev => Math.min(totalFrames - 1, prev + 1))}
              className="p-2 rounded-full bg-white/20 hover:bg-white/30 text-white transition-colors"
              disabled={selectedFrame === totalFrames - 1}
            >
              <ChevronRight size={16} />
            </button>
            
            <input
              type="range"
              min="0"
              max={totalFrames - 1}
              value={selectedFrame}
              onChange={(e) => setSelectedFrame(parseInt(e.target.value))}
              className="w-32 md:w-64"
            />
          </div>
          
          <div className="flex items-center space-x-2">
            <button 
              onClick={() => setZoomLevel(prev => Math.max(0.5, prev - 0.1))}
              className="p-2 rounded-full bg-white/20 hover:bg-white/30 text-white transition-colors"
              disabled={zoomLevel <= 0.5}
            >
              <ZoomOut size={16} />
            </button>
            
            <button 
              onClick={() => setZoomLevel(prev => Math.min(2, prev + 0.1))}
              className="p-2 rounded-full bg-white/20 hover:bg-white/30 text-white transition-colors"
              disabled={zoomLevel >= 2}
            >
              <ZoomIn size={16} />
            </button>
            
            <button 
              className="p-2 rounded-full bg-white/20 hover:bg-white/30 text-white transition-colors"
              title="Download frame"
            >
              <Download size={16} />
            </button>
            
            <button 
              className="p-2 rounded-full bg-white/20 hover:bg-white/30 text-white transition-colors"
              title="Share"
            >
              <Share2 size={16} />
            </button>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="space-y-12">
      {/* Hero Section */}
      <motion.section 
        className="text-center py-8"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
      >
        <h1 className="text-4xl md:text-5xl font-bold mb-4 bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
          Advanced Pose Detection Platform
        </h1>
        <p className="text-xl text-gray-600 dark:text-gray-300 max-w-3xl mx-auto mb-8">
          Extract human joint positions from videos and images with state-of-the-art accuracy using AlphaPose technology
        </p>
        <div className="flex flex-wrap justify-center gap-4">
          <Link 
            href="#upload"
            className="inline-flex items-center justify-center px-6 py-3 border border-transparent text-base font-medium rounded-md text-white bg-blue-600 hover:bg-blue-700 transition-colors"
          >
            Start Detecting <ArrowRight className="ml-2 h-5 w-5" />
          </Link>
          <Link 
            href="/docs"
            className="inline-flex items-center justify-center px-6 py-3 border border-gray-300 dark:border-gray-600 text-base font-medium rounded-md text-gray-700 dark:text-gray-200 bg-white dark:bg-gray-800 hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors"
          >
            Learn More
          </Link>
        </div>
      </motion.section>

      {/* Key Features */}
      <section className="py-8">
        <h2 className="text-3xl font-bold text-center mb-8">Key Features</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          {features.map((feature, index) => (
            <motion.div
              key={index}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: index * 0.1 }}
              className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 hover:shadow-md transition-shadow"
            >
              <div className="bg-blue-50 dark:bg-blue-900/20 p-3 rounded-full w-12 h-12 flex items-center justify-center mb-4">
                {feature.icon}
              </div>
              <h3 className="text-xl font-bold mb-2">{feature.title}</h3>
              <p className="text-gray-600 dark:text-gray-400">{feature.description}</p>
            </motion.div>
          ))}
        </div>
      </section>

      {/* Upload Area */}
      <section id="upload" className="py-8">
        <h2 className="text-3xl font-bold text-center mb-8">Upload Media</h2>
        <div className="max-w-3xl mx-auto">
          <div
            {...getRootProps()}
            className={`border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-colors ${
              isDragActive 
                ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20' 
                : 'border-gray-300 hover:border-blue-400 dark:border-gray-600'
            }`}
          >
            <input {...getInputProps()} />
            <div className="space-y-4">
              <Upload className="h-12 w-12 mx-auto text-gray-400 dark:text-gray-500" />
              {uploadState.isUploading ? (
                <div className="space-y-4">
                  <p className="text-lg font-medium">
                    Uploading {uploadState.file?.name}...
                  </p>
                  <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2.5">
                    <div 
                      className="bg-blue-600 h-2.5 rounded-full transition-all duration-300"
                      style={{ width: `${uploadState.progress}%` }}
                    ></div>
                  </div>
                </div>
              ) : (
                <>
                  <p className="text-lg font-medium">
                    {isDragActive
                      ? "Drop the files here"
                      : "Drag & drop files here or click to browse"}
                  </p>
                  <p className="text-sm text-gray-500 dark:text-gray-400">
                    Support for video files (MP4, AVI, MOV) and images (JPG, PNG)
                  </p>
                </>
              )}
            </div>
          </div>
          
          {acceptedFiles.length > 0 && !uploadState.isUploading && (
            <div className="mt-4 p-4 bg-gray-50 dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700">
              <p className="font-medium">Selected file:</p>
              <p className="text-sm text-gray-600 dark:text-gray-300">
                {acceptedFiles[0].name} ({(acceptedFiles[0].size / (1024 * 1024)).toFixed(2)} MB)
              </p>
              <div className="mt-2">
                <button 
                  onClick={() => handleFileUpload(acceptedFiles[0])}
                  disabled={isLoading}
                  className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                >
                  Process File
                </button>
              </div>
            </div>
          )}
        </div>
      </section>

      {/* Quick Start Guide */}
      <section className="py-8 bg-gray-50 dark:bg-gray-800/50 -mx-6 px-6 rounded-lg">
        <h2 className="text-3xl font-bold text-center mb-8">Quick Start Guide</h2>
        <div className="max-w-4xl mx-auto">
          {quickStartSteps.map((step, index) => (
            <motion.div
              key={index}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.5, delay: index * 0.2 }}
              className="relative pl-12 pb-8"
            >
              {/* Step number circle */}
              <div className="absolute left-0 top-0 bg-blue-600 text-white w-8 h-8 rounded-full flex items-center justify-center font-bold">
                {step.number}
              </div>
              
              {/* Connecting line */}
              {index < quickStartSteps.length - 1 && (
                <div className="absolute left-4 top-8 bottom-0 w-0.5 bg-blue-200 dark:bg-blue-800"></div>
              )}
              
              <h3 className="text-xl font-bold mb-2">{step.title}</h3>
              <p className="text-gray-600 dark:text-gray-400">{step.description}</p>
            </motion.div>
          ))}
        </div>
      </section>

      {/* Statistics Dashboard */}
      <section className="py-8">
        <h2 className="text-3xl font-bold text-center mb-8">Platform Statistics</h2>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
          <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700">
            <div className="flex items-start justify-between">
              <div>
                <p className="text-sm font-medium text-gray-500 dark:text-gray-400">Total Tasks</p>
                <p className="text-3xl font-bold mt-1">{stats.totalTasks}</p>
              </div>
              <div className="bg-blue-50 dark:bg-blue-900/20 p-2 rounded-md">
                <Layers className="h-6 w-6 text-blue-500" />
              </div>
            </div>
            {stats.totalTasks > 0 && (
              <div className="mt-4 flex items-center text-xs text-green-600 dark:text-green-400">
                <span className="flex items-center">
                  <ArrowRight className="h-3 w-3 mr-1 rotate-45" />
                  +10% from last week
                </span>
              </div>
            )}
          </div>
          
          <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700">
            <div className="flex items-start justify-between">
              <div>
                <p className="text-sm font-medium text-gray-500 dark:text-gray-400">Successful Tasks</p>
                <p className="text-3xl font-bold mt-1">{stats.successfulTasks}</p>
              </div>
              <div className="bg-green-50 dark:bg-green-900/20 p-2 rounded-md">
                <Video className="h-6 w-6 text-green-500" />
              </div>
            </div>
            {stats.successfulTasks > 0 && (
              <div className="mt-4 flex items-center text-xs text-green-600 dark:text-green-400">
                <span className="flex items-center">
                  <ArrowRight className="h-3 w-3 mr-1 rotate-45" />
                  +5% from last week
                </span>
              </div>
            )}
          </div>
          
          <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700">
            <div className="flex items-start justify-between">
              <div>
                <p className="text-sm font-medium text-gray-500 dark:text-gray-400">Avg. Processing Time</p>
                <p className="text-3xl font-bold mt-1">{stats.processingTime > 0 ? `${stats.processingTime.toFixed(1)}s` : 'N/A'}</p>
              </div>
              <div className="bg-yellow-50 dark:bg-yellow-900/20 p-2 rounded-md">
                <Zap className="h-6 w-6 text-yellow-500" />
              </div>
            </div>
          </div>
          
          <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700">
            <div className="flex items-start justify-between">
              <div>
                <p className="text-sm font-medium text-gray-500 dark:text-gray-400">Detected Poses</p>
                <p className="text-3xl font-bold mt-1">{stats.detectedPoses}</p>
              </div>
              <div className="bg-purple-50 dark:bg-purple-900/20 p-2 rounded-md">
                <ImageIcon className="h-6 w-6 text-purple-500" />
              </div>
            </div>
            {stats.detectedPoses > 0 && (
              <div className="mt-4 flex items-center text-xs text-green-600 dark:text-green-400">
                <span className="flex items-center">
                  <ArrowRight className="h-3 w-3 mr-1 rotate-45" />
                  +15% from last week
                </span>
              </div>
            )}
          </div>
        </div>
      </section>

      {/* Recent Tasks with Enhanced Visualization */}
      <section className="py-8">
        <div className="flex justify-between items-center mb-6">
          <h2 className="text-3xl font-bold">Recent Tasks</h2>
          <Link 
            href="/tasks" 
            className="text-blue-600 dark:text-blue-400 hover:underline flex items-center"
          >
            View All <ArrowRight className="ml-1 h-4 w-4" />
          </Link>
        </div>
        
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Task list */}
          <div className="lg:col-span-1 space-y-4 max-h-[600px] overflow-y-auto pr-2">
            <AnimatePresence>
              {tasks.length > 0 ? (
                tasks.map((task, index) => (
                  <motion.div
                    key={task.id}
                    initial={{ opacity: 0, scale: 0.9 }}
                    animate={{ opacity: 1, scale: 1 }}
                    transition={{ duration: 0.3, delay: index * 0.1 }}
                    onClick={() => setSelectedTask(task.id)}
                    className={`p-4 rounded-lg border cursor-pointer transition-all ${
                      selectedTask === task.id 
                        ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20 shadow-md' 
                        : 'border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 hover:border-blue-300 dark:hover:border-blue-700'
                    }`}
                  >
                    <div className="flex items-center justify-between">
                      <h3 className="font-medium truncate" title={task.filename}>
                        {task.filename}
                      </h3>
                      <span className={`text-xs px-2 py-1 rounded-full ${
                        task.status === 'SUCCESS' ? 'bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-300' :
                        task.status === 'RUNNING' ? 'bg-blue-100 text-blue-800 dark:bg-blue-900/30 dark:text-blue-300' :
                        task.status === 'FAILED' ? 'bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-300' :
                        'bg-gray-100 text-gray-800 dark:bg-gray-700 dark:text-gray-300'
                      }`}>
                        {task.status}
                      </span>
                    </div>
                    
                    <div className="mt-2 text-xs text-gray-500 dark:text-gray-400">
                      {task.created_at && (
                        <p>Created {formatDistanceToNow(new Date(task.created_at))} ago</p>
                      )}
                    </div>
                    
                    {task.status === 'RUNNING' && (
                      <div className="mt-2">
                        <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-1.5">
                          <div 
                            className="bg-blue-600 h-1.5 rounded-full"
                            style={{ width: '60%' }}
                          ></div>
                        </div>
                      </div>
                    )}
                  </motion.div>
                ))
              ) : (
                <motion.div
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  className="text-center py-12 text-gray-500 dark:text-gray-400 bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700"
                >
                  <p className="text-lg">No tasks yet. Upload a video or image to get started.</p>
                </motion.div>
              )}
            </AnimatePresence>
          </div>
          
          {/* Visualization and feedback panel */}
          <div className="lg:col-span-2 space-y-6">
            {selectedTask ? (
              <>
                {/* Pose visualization */}
                <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border border-gray-200 dark:border-gray-700">
                  <h3 className="text-xl font-bold mb-4">Pose Visualization</h3>
                  <PoseVisualization 
                    taskId={selectedTask}
                    frameIndex={selectedFrame}
                    totalFrames={30} // This would come from the task data
                  />
                </div>
                
                {/* User feedback section */}
                <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border border-gray-200 dark:border-gray-700">
                  <h3 className="text-xl font-bold mb-4">Provide Feedback</h3>
                  
                  {/* Rating buttons */}
                  <div className="flex items-center space-x-4 mb-4">
                    <button className="flex items-center space-x-2 px-3 py-2 rounded-md bg-green-100 dark:bg-green-900/30 text-green-800 dark:text-green-300 hover:bg-green-200 dark:hover:bg-green-900/50 transition-colors">
                      <ThumbsUp size={16} />
                      <span>Good Detection</span>
                    </button>
                    
                    <button className="flex items-center space-x-2 px-3 py-2 rounded-md bg-red-100 dark:bg-red-900/30 text-red-800 dark:text-red-300 hover:bg-red-200 dark:hover:bg-red-900/50 transition-colors">
                      <ThumbsDown size={16} />
                      <span>Needs Improvement</span>
                    </button>
                  </div>
                  
                  {/* Tags */}
                  <div className="mb-4">
                    <label className="block text-sm font-medium mb-2">
                      Select issues or tags:
                    </label>
                    <div className="flex flex-wrap gap-2">
                      {availableTags.map(tag => (
                        <button
                          key={tag}
                          onClick={() => toggleTag(tag)}
                          className={`flex items-center space-x-1 px-2 py-1 rounded-md text-sm ${
                            selectedTags.includes(tag)
                              ? 'bg-blue-100 text-blue-800 dark:bg-blue-900/50 dark:text-blue-300'
                              : 'bg-gray-100 text-gray-800 dark:bg-gray-700 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-600'
                          }`}
                        >
                          <Tag size={12} />
                          <span>{tag}</span>
                        </button>
                      ))}
                    </div>
                  </div>
                  
                  {/* Comment */}
                  <div className="mb-4">
                    <label className="block text-sm font-medium mb-2">
                      Additional comments:
                    </label>
                    <textarea
                      value={currentComment}
                      onChange={(e) => setCurrentComment(e.target.value)}
                      className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
                      rows={3}
                      placeholder="Describe any issues or observations..."
                    ></textarea>
                  </div>
                  
                  {/* Submit button */}
                  <button
                    onClick={handleFeedbackSubmit}
                    disabled={selectedTags.length === 0}
                    className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                  >
                    <MessageSquare className="mr-2 h-4 w-4" />
                    Submit Feedback
                  </button>
                </div>
              </>
            ) : (
              <div className="bg-white dark:bg-gray-800 p-6 rounded-lg border border-gray-200 dark:border-gray-700 text-center">
                <div className="py-12">
                  <ImageIcon className="h-16 w-16 mx-auto text-gray-400 dark:text-gray-500 mb-4" />
                  <h3 className="text-xl font-medium mb-2">No task selected</h3>
                  <p className="text-gray-500 dark:text-gray-400">
                    Select a task from the list to view pose detection results
                  </p>
                </div>
              </div>
            )}
          </div>
        </div>
      </section>

      {/* Call to Action */}
      <section className="py-12 bg-gradient-to-r from-blue-600 to-purple-600 -mx-6 px-6 rounded-lg text-white">
        <div className="text-center max-w-3xl mx-auto py-8">
          <h2 className="text-3xl font-bold mb-4">Ready to Start?</h2>
          <p className="text-xl mb-8 text-blue-100">
            Transform your videos and images into detailed pose data with just a few clicks
          </p>
          <div className="flex flex-wrap justify-center gap-4">
            <Link 
              href="#upload"
              className="inline-flex items-center justify-center px-6 py-3 border border-transparent text-base font-medium rounded-md text-blue-700 bg-white hover:bg-gray-100 transition-colors"
            >
              Upload Media
            </Link>
            <Link 
              href="/projects/new"
              className="inline-flex items-center justify-center px-6 py-3 border border-white text-base font-medium rounded-md text-white bg-transparent hover:bg-white/10 transition-colors"
            >
              Create Project
            </Link>
          </div>
        </div>
      </section>
    </div>
  )
}
