'use client'

import React, { useMemo } from 'react'
import Link from 'next/link'
import { usePathname } from 'next/navigation'
import { Home, ChevronRight } from 'lucide-react'

/**
 * Breadcrumbs component that shows the current page path with navigation links
 * Automatically generates breadcrumbs based on the current URL path
 */
const Breadcrumbs: React.FC = () => {
  const pathname = usePathname()

  // Generate breadcrumb items from the pathname
  const breadcrumbs = useMemo(() => {
    // Skip rendering breadcrumbs on home page
    if (pathname === '/') return []

    // Split the pathname into segments and filter out empty segments
    const segments = pathname.split('/').filter(segment => segment)
    
    // Create breadcrumb items with paths
    return segments.map((segment, index) => {
      // Build the path for this breadcrumb
      const path = `/${segments.slice(0, index + 1).join('/')}`
      
      // Format the label (capitalize and replace hyphens with spaces)
      const label = segment
        .replace(/-/g, ' ')
        .replace(/\b\w/g, char => char.toUpperCase())
      
      // For dynamic routes with IDs, try to make them more readable
      const displayLabel = segment.match(/^[0-9a-f]{8,}$/) 
        ? 'Details' 
        : label
      
      return {
        label: displayLabel,
        path,
        isLast: index === segments.length - 1
      }
    })
  }, [pathname])

  // Don't render breadcrumbs on home page
  if (breadcrumbs.length === 0) {
    return null
  }

  return (
    <nav aria-label="Breadcrumb">
      <ol className="flex items-center flex-wrap space-x-2 text-sm">
        {/* Home link is always first */}
        <li>
          <Link 
            href="/" 
            className="flex items-center text-gray-500 hover:text-blue-600 dark:text-gray-400 dark:hover:text-blue-400 transition-colors"
            aria-label="Home"
          >
            <Home className="h-4 w-4" />
          </Link>
        </li>
        
        {/* Separator after home */}
        <li className="text-gray-400 dark:text-gray-600">
          <ChevronRight className="h-4 w-4" />
        </li>
        
        {/* Render each breadcrumb item */}
        {breadcrumbs.map((breadcrumb, index) => (
          <React.Fragment key={breadcrumb.path}>
            <li>
              {breadcrumb.isLast ? (
                // Current page (not clickable)
                <span className="font-medium text-gray-800 dark:text-gray-200" aria-current="page">
                  {breadcrumb.label}
                </span>
              ) : (
                // Link to previous path
                <Link
                  href={breadcrumb.path}
                  className="text-gray-500 hover:text-blue-600 dark:text-gray-400 dark:hover:text-blue-400 transition-colors"
                >
                  {breadcrumb.label}
                </Link>
              )}
            </li>
            
            {/* Add separator between items (but not after the last one) */}
            {!breadcrumb.isLast && (
              <li className="text-gray-400 dark:text-gray-600">
                <ChevronRight className="h-4 w-4" />
              </li>
            )}
          </React.Fragment>
        ))}
      </ol>
    </nav>
  )
}

export default Breadcrumbs
