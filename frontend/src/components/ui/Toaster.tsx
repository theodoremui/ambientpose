'use client'

import React, { useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { useNotifications } from '@/app/providers'
import { 
  CheckCircle, 
  XCircle, 
  AlertTriangle, 
  Info, 
  X 
} from 'lucide-react'

/**
 * Toast notification component that displays notifications from the notification context
 * Supports different types (success, error, warning, info) with appropriate styling and animations
 */
export const Toaster: React.FC = () => {
  const { notifications, removeNotification } = useNotifications()

  return (
    <div className="fixed top-4 right-4 z-50 flex flex-col gap-2 w-full max-w-md">
      <AnimatePresence>
        {notifications.map((notification) => (
          <motion.div
            key={notification.id}
            initial={{ opacity: 0, y: -20, scale: 0.95 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: -20, scale: 0.95 }}
            transition={{ duration: 0.2 }}
            className={`
              flex items-start p-4 rounded-lg shadow-lg border-l-4
              ${getNotificationStyles(notification.type)}
            `}
          >
            <div className="flex-shrink-0 mr-3">
              {getNotificationIcon(notification.type)}
            </div>
            <div className="flex-1 mr-2">
              <h4 className="text-sm font-medium">
                {notification.title}
              </h4>
              <p className="text-sm mt-1 opacity-90">
                {notification.message}
              </p>
            </div>
            <button
              onClick={() => removeNotification(notification.id)}
              className="flex-shrink-0 text-gray-400 hover:text-gray-600 dark:text-gray-500 dark:hover:text-gray-300 transition-colors"
              aria-label="Close notification"
            >
              <X size={18} />
            </button>
          </motion.div>
        ))}
      </AnimatePresence>
    </div>
  )
}

/**
 * Get the appropriate icon based on notification type
 */
const getNotificationIcon = (type: 'success' | 'error' | 'warning' | 'info') => {
  switch (type) {
    case 'success':
      return <CheckCircle className="h-5 w-5 text-green-500" />
    case 'error':
      return <XCircle className="h-5 w-5 text-red-500" />
    case 'warning':
      return <AlertTriangle className="h-5 w-5 text-amber-500" />
    case 'info':
      return <Info className="h-5 w-5 text-blue-500" />
    default:
      return <Info className="h-5 w-5 text-blue-500" />
  }
}

/**
 * Get the appropriate styles based on notification type
 */
const getNotificationStyles = (type: 'success' | 'error' | 'warning' | 'info') => {
  switch (type) {
    case 'success':
      return 'bg-white dark:bg-gray-800 border-green-500 text-gray-800 dark:text-gray-200'
    case 'error':
      return 'bg-white dark:bg-gray-800 border-red-500 text-gray-800 dark:text-gray-200'
    case 'warning':
      return 'bg-white dark:bg-gray-800 border-amber-500 text-gray-800 dark:text-gray-200'
    case 'info':
      return 'bg-white dark:bg-gray-800 border-blue-500 text-gray-800 dark:text-gray-200'
    default:
      return 'bg-white dark:bg-gray-800 border-blue-500 text-gray-800 dark:text-gray-200'
  }
}

export default Toaster
